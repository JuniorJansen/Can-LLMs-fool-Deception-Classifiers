import argparse
import json
import os
import random
import statistics
from tqdm import tqdm

from src.model.classifier import DeceptionClassifier
from src.data.load_data import load_dataset
from src.config import DATA_PATH, MAX_ITER, NUM_CANDIDATES, SIMILARITY_THRESHOLD
from src.attack.paraphraser import Paraphraser
from src.attack.attack_loop import run_attack


STRATEGIES = ["zero_shot", "one_shot", "few_shot"]
MODELS = ["llama3.2", "mistral", "gemma2:2b", "qwen2.5:7b"]
N = 100
SEED = 42


def sample_balanced(texts, labels, n, seed=SEED):
    """Sample n texts with equal class balance (n/2 per class)."""
    rng = random.Random(seed)
    per_class = n // 2

    class_0 = [(t, l) for t, l in zip(texts, labels) if l == 0]
    class_1 = [(t, l) for t, l in zip(texts, labels) if l == 1]

    sampled_0 = rng.sample(class_0, min(per_class, len(class_0)))
    sampled_1 = rng.sample(class_1, min(per_class, len(class_1)))

    combined = sampled_0 + sampled_1
    rng.shuffle(combined)

    texts_out, labels_out = zip(*combined)
    return list(texts_out), list(labels_out)


def compute_summary(results):
    """Compute rich summary statistics from a list of per-sample results."""
    n = len(results)
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    asr = len(successes) / n

    conf_changes = [r["max_confidence_change"] for r in results]
    similarities = [r["final_similarity"] for r in results]
    iterations = [r["iterations"] for r in results]

    # Per-class breakdown (original label)
    class_0 = [r for r in results if r["original_label"] == 0]  # truthful
    class_1 = [r for r in results if r["original_label"] == 1]  # deceptive

    def safe_stats(values):
        if not values:
            return {"mean": None, "median": None, "std": None, "min": None, "max": None}
        return {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    def class_summary(subset):
        if not subset:
            return {"n": 0, "asr": None}
        sub_successes = [r for r in subset if r["success"]]
        return {
            "n": len(subset),
            "asr": round(len(sub_successes) / len(subset), 4),
            "avg_confidence_change": round(statistics.mean([r["max_confidence_change"] for r in subset]), 4),
            "avg_similarity": round(statistics.mean([r["final_similarity"] for r in subset]), 4),
        }

    summary = {
        "n_samples": n,
        "attack_success_rate": round(asr, 4),
        "confidence_change": safe_stats(conf_changes),
        "semantic_similarity": safe_stats(similarities),
        "iterations": safe_stats([float(v) for v in iterations]),
        "successful_attacks": {
            "count": len(successes),
            "avg_iterations": round(statistics.mean([r["iterations"] for r in successes]), 4) if successes else None,
            "avg_similarity": round(statistics.mean([r["final_similarity"] for r in successes]), 4) if successes else None,
            "avg_confidence_change": round(statistics.mean([r["max_confidence_change"] for r in successes]), 4) if successes else None,
        },
        "failed_attacks": {
            "count": len(failures),
            "avg_confidence_change": round(statistics.mean([r["max_confidence_change"] for r in failures]), 4) if failures else None,
        },
        "per_class": {
            "truthful": class_summary(class_0),
            "deceptive": class_summary(class_1),
        },
    }
    return summary


def run_experiment(texts, labels, classifier, model, strategy):
    print(f"\n{'='*50}")
    print(f"Model: {model} | Strategy: {strategy}")
    print(f"{'='*50}")

    paraphraser = Paraphraser(model=model, strategy=strategy)

    results = []
    success = 0
    n = len(texts)

    pbar = tqdm(zip(texts, labels), total=n, desc=f"Attacking ({strategy})")

    for i, (text, true_label) in enumerate(pbar):
        result = run_attack(text, classifier, paraphraser)

        if result["success"]:
            success += 1

        results.append({
            "sample_index": i,
            "original_text": text,
            "true_label": true_label,
            "attacked_text": result["final_text"],
            "original_label": result["original_label"],
            "new_label": result["new_label"],
            "original_confidence": result["original_confidence"],
            "final_confidence": result["final_confidence"],
            "success": result["success"],
            "iterations": result["iterations"],
            "confidence_change": result["confidence_change"],
            "max_confidence_change": result["max_confidence_change"],
            "final_similarity": result["final_similarity"],
            "iteration_log": result["iteration_log"],
        })
        pbar.set_postfix({"ASR": round(success / (i + 1), 3)})

    summary = compute_summary(results)
    print(f"\nASR: {summary['attack_success_rate']} | "
          f"Avg conf change: {summary['confidence_change']['mean']} | "
          f"Avg similarity: {summary['semantic_similarity']['mean']}")

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Run adversarial paraphrase attacks")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model to run (default: all). Choose from: {MODELS}")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODELS

    print("Loading dataset...")
    all_texts, all_labels = load_dataset(DATA_PATH)
    print(f"Dataset size: {len(all_texts)}")

    print(f"\nSampling {N} balanced samples ({N//2} per class, seed={SEED})...")
    texts, labels = sample_balanced(all_texts, all_labels, N)
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    print(f"Class distribution: truthful={class_counts[0]}, deceptive={class_counts[1]}")

    print("\nLoading classifier...")
    classifier = DeceptionClassifier()

    # ===== BASELINE =====
    print("\nComputing baseline accuracy...")
    correct = sum(
        1 for text, label in tqdm(zip(texts, labels), total=N, desc="Baseline")
        if classifier.predict(text)[0] == label
    )
    baseline_acc = correct / N
    print("Baseline accuracy:", round(baseline_acc, 3))

    os.makedirs("results", exist_ok=True)

    # ===== RUN MODELS =====
    for model in models_to_run:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model}")
        print(f"{'#'*60}")

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
            results, summary = run_experiment(texts, labels, classifier, model, strategy)

            file_path = f"results/{model}_{strategy}.json"
            with open(file_path, "w") as f:
                json.dump(results, f, indent=2)
            print("Saved to:", os.path.abspath(file_path))

            overall_summary["strategies"][strategy] = summary

        # ===== SAVE SUMMARY =====
        summary_path = f"results/{model}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(overall_summary, f, indent=2)
        print(f"\nSummary saved to:", os.path.abspath(summary_path))


if __name__ == "__main__":
    main()
