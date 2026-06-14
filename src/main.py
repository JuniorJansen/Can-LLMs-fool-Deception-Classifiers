"""Run adversarial paraphrase attacks against the DistilBERT deception classifier.

For each (attacker model, prompting strategy) condition, the script:
  1. Draws a balanced 200-narrative sample from the HippoCorpus test split.
  2. Runs the iterative paraphrase attack defined in ``src.attack.attack_loop``.
  3. Writes a per-sample JSON file and a per-model summary JSON to ``results/``.

Resumable: the per-sample loop checkpoints after every example, so interrupted
runs pick up where they stopped. Strategies whose final result file already
exists are skipped.

Usage:
    python -m src.main --model llama3.2
    python -m src.main --model openai/gpt-oss-120b
    python -m src.main                       # run every model in sequence
"""

import argparse
import json
import os
import random
import statistics

from tqdm import tqdm

from src.attack.attack_loop import run_attack
from src.attack.paraphraser import FEW_SHOT_EXAMPLES, ONE_SHOT_EXAMPLES, Paraphraser
from src.config import (
    DATA_PATH,
    MAX_ITER,
    NUM_CANDIDATES,
    SEED,
    SIMILARITY_THRESHOLD,
    set_seed,
)
from src.data.load_data import load_dataset
from src.model.classifier import DeceptionClassifier

STRATEGIES = ["zero_shot", "one_shot", "few_shot"]

# Attacker models, in the order they are reported in the thesis.
# Names match those expected by Ollama (smaller tier) and the hosted endpoints in
# PROVIDERS (larger tier). On disk, the ":" and "/" characters get replaced with
# "_" so each model produces files like ``gemma3_4b_zero_shot.json``.
MODELS = [
    "llama3.2",
    "mistral",
    "gemma3:4b",
    "qwen2.5:7b",
    "meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-oss-120b",
]

N = 200  # balanced sample size, matched to the thesis


def _extract_example_texts():
    """Pull the originals out of the few-shot prompt examples so we can exclude
    them from the test sample (otherwise the LLM is asked to attack a narrative
    it has already seen as a worked example)."""
    texts = set()
    for source in list(ONE_SHOT_EXAMPLES.values()) + list(FEW_SHOT_EXAMPLES.values()):
        for line in source.split("\n"):
            if line.startswith("Original"):
                start = line.index('"') + 1
                end = line.rindex('"')
                texts.add(line[start:end])
    return texts


EXAMPLE_TEXTS = _extract_example_texts()


def sample_balanced(texts, labels, n, seed=SEED):
    """Sample n texts with equal class balance (n/2 per class)."""
    rng = random.Random(seed)
    per_class = n // 2
    class_0 = [(t, l) for t, l in zip(texts, labels) if l == 0]
    class_1 = [(t, l) for t, l in zip(texts, labels) if l == 1]
    sampled = rng.sample(class_0, per_class) + rng.sample(class_1, per_class)
    rng.shuffle(sampled)
    out_texts, out_labels = zip(*sampled)
    return list(out_texts), list(out_labels)


def _safe_stats(values):
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _class_summary(subset):
    if not subset:
        return {"n": 0, "asr": None}
    successes = [r for r in subset if r["success"]]
    return {
        "n": len(subset),
        "asr": round(len(successes) / len(subset), 4),
        "avg_confidence_change": round(statistics.mean(r["max_confidence_change"] for r in subset), 4),
        "avg_similarity": round(statistics.mean(r["final_similarity"] for r in subset), 4),
    }


def compute_summary(results):
    """Aggregate per-sample results into the headline metrics for one condition."""
    n = len(results)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    class_truthful = [r for r in results if r["original_label"] == 0]
    class_deceptive = [r for r in results if r["original_label"] == 1]

    return {
        "n_samples": n,
        "attack_success_rate": round(len(successes) / n, 4),
        "confidence_change": _safe_stats([r["max_confidence_change"] for r in results]),
        "semantic_similarity": _safe_stats([r["final_similarity"] for r in results]),
        "iterations": _safe_stats([float(r["iterations"]) for r in results]),
        "successful_attacks": {
            "count": len(successes),
            "avg_iterations": round(statistics.mean(r["iterations"] for r in successes), 4) if successes else None,
            "avg_similarity": round(statistics.mean(r["final_similarity"] for r in successes), 4) if successes else None,
            "avg_confidence_change": round(statistics.mean(r["max_confidence_change"] for r in successes), 4) if successes else None,
        },
        "failed_attacks": {
            "count": len(failures),
            "avg_confidence_change": round(statistics.mean(r["max_confidence_change"] for r in failures), 4) if failures else None,
        },
        "per_class": {
            "truthful": _class_summary(class_truthful),
            "deceptive": _class_summary(class_deceptive),
        },
    }


def run_experiment(texts, labels, classifier, model, strategy, model_safe):
    print(f"\n{'=' * 50}\nModel: {model} | Strategy: {strategy}\n{'=' * 50}")
    checkpoint_path = f"results/.checkpoint_{model_safe}_{strategy}.json"

    results = []
    start_index = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt["results"]
        start_index = ckpt["last_completed_index"] + 1
        print(f"Resuming from sample {start_index} (checkpoint found)")

    paraphraser = Paraphraser(model=model, strategy=strategy)
    success = sum(r["success"] for r in results)
    n = len(texts)

    pbar = tqdm(zip(texts[start_index:], labels[start_index:]),
                total=n, initial=start_index, desc=f"Attacking ({strategy})")
    for i, (text, true_label) in enumerate(pbar, start=start_index):
        result = run_attack(text, classifier, paraphraser)
        if result["success"]:
            success += 1
        results.append({
            "sample_index": i,
            "original_text": text,
            "true_label": true_label,
            "attacked_text": result["final_text"],
            **{k: result[k] for k in (
                "original_label", "new_label", "original_confidence", "final_confidence",
                "success", "iterations", "confidence_change", "max_confidence_change",
                "final_similarity", "iteration_log",
            )},
        })
        # Checkpoint after every example so a crashed run loses at most one sample.
        with open(checkpoint_path, "w") as f:
            json.dump({"last_completed_index": i, "results": results}, f)
        pbar.set_postfix({"ASR": round(success / (i + 1), 3)})

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    summary = compute_summary(results)
    print(f"\nASR: {summary['attack_success_rate']} | "
          f"Avg conf change: {summary['confidence_change']['mean']} | "
          f"Avg similarity: {summary['semantic_similarity']['mean']}")
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Run adversarial paraphrase attacks")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model to run (default: every model in MODELS). Choose from: {MODELS}")
    args = parser.parse_args()

    set_seed()
    models_to_run = [args.model] if args.model else MODELS

    print("Loading dataset...")
    all_texts, all_labels = load_dataset(DATA_PATH)
    print(f"Dataset size: {len(all_texts)}")

    # Drop any narrative used inside an in-context example, to prevent data leakage.
    filtered = [(t, l) for t, l in zip(all_texts, all_labels) if t not in EXAMPLE_TEXTS]
    excluded = len(all_texts) - len(filtered)
    if excluded:
        print(f"Excluded {excluded} example text(s) from test set to prevent data leakage.")
    filtered_texts, filtered_labels = zip(*filtered)

    print(f"\nSampling {N} balanced samples ({N // 2} per class, seed={SEED})...")
    texts, labels = sample_balanced(list(filtered_texts), list(filtered_labels), N)
    print(f"Class distribution: truthful={labels.count(0)}, deceptive={labels.count(1)}")

    print("\nLoading classifier...")
    classifier = DeceptionClassifier()

    print("\nComputing baseline accuracy...")
    correct = sum(
        1 for text, label in tqdm(zip(texts, labels), total=N, desc="Baseline")
        if classifier.predict(text)[0] == label
    )
    baseline_acc = correct / N
    print("Baseline accuracy:", round(baseline_acc, 3))

    os.makedirs("results", exist_ok=True)

    for model in models_to_run:
        print(f"\n{'#' * 60}\n# MODEL: {model}\n{'#' * 60}")
        model_safe = model.replace(":", "_").replace("/", "_")

        overall_summary = {
            "model": model,
            "n_samples": N,
            "max_iterations": MAX_ITER,
            "num_candidates": NUM_CANDIDATES,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "baseline_accuracy": baseline_acc,
            "strategies": {},
        }

        for strategy in STRATEGIES:
            file_path = f"results/{model_safe}_{strategy}.json"
            checkpoint_path = f"results/.checkpoint_{model_safe}_{strategy}.json"

            # If the final file is already there and no checkpoint is sitting next
            # to it, the strategy ran to completion last time; just reload its summary.
            if os.path.exists(file_path) and not os.path.exists(checkpoint_path):
                print(f"\nSkipping {model} | {strategy}: results already at {file_path}")
                with open(file_path) as f:
                    results = json.load(f)
                overall_summary["strategies"][strategy] = compute_summary(results)
                continue

            results, summary = run_experiment(texts, labels, classifier, model, strategy, model_safe)
            with open(file_path, "w") as f:
                json.dump(results, f, indent=2)
            print("Saved to:", os.path.abspath(file_path))
            overall_summary["strategies"][strategy] = summary

        summary_path = f"results/{model_safe}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(overall_summary, f, indent=2)
        print("\nSummary saved to:", os.path.abspath(summary_path))


if __name__ == "__main__":
    main()
