"""
Generate and validate one-shot / few-shot example paraphrases using a neutral model.

Usage:
    python -m src.generate_examples --model <ollama_model> [--candidates N] [--iters N] [--min-conf-change F]

The script iterates over multiple rounds of generation, tracks the best paraphrase
per source text (highest confidence change that also flips the label), and only
accepts it if the confidence change exceeds --min-conf-change. This ensures the
example paraphrases are strong, not just barely successful.
"""

import argparse
import textwrap

from src.model.classifier import DeceptionClassifier
from src.attack.paraphraser import Paraphraser
from src.attack.similarity import compute_similarity

# ---------------------------------------------------------------------------
# Source texts to generate examples for.
# Label: 0 = truthful, 1 = deceptive.
# ---------------------------------------------------------------------------
SOURCE_TEXTS = [
    # Only need one more deceptive_to_truthful example (examples 1 and 2 already done)
    {
        "direction": "deceptive_to_truthful",
        "original_label": 1,
        "text": (
            "About five months ago, I wanted to move in a different professional direction. "
            "I decided driving trucks would be great. I went ahead and tried to pass the CDL, but didn't. "
            "I guess I wasn't ready. I went home and did some practice. I read up on tips on how to pass. "
            "I tried again the following week and passed. I was so happy. I got my first job offer "
            "the following day."
        ),
    },
]

SIMILARITY_THRESHOLD = 0.85


def find_best_paraphrase(source, classifier, paraphraser, n_candidates=10, max_iters=10, min_conf_change=0.3):
    """
    Iterate up to max_iters rounds of generation, tracking the candidate with the
    highest confidence change that also flips the classifier's prediction and meets
    the similarity threshold. Returns the best candidate only if its confidence
    change exceeds min_conf_change.
    """
    original_label = source["original_label"]
    text = source["text"]
    target_label = 1 - original_label

    pred, original_conf = classifier.predict(text)
    if pred != original_label:
        print("  [WARN] Classifier already disagrees with the stated label — skipping.")
        return None

    best = None

    for i in range(max_iters):
        raw = paraphraser.generate(text, k=n_candidates, original_label=original_label)
        for candidate in raw:
            if candidate == text:
                continue
            sim = compute_similarity(text, candidate)
            if sim < SIMILARITY_THRESHOLD:
                continue
            pred_c, conf_c = classifier.predict(candidate)
            if pred_c != target_label:
                continue
            conf_change = abs(conf_c - original_conf)
            if best is None or conf_change > best["conf_change"]:
                best = {
                    "text": candidate,
                    "similarity": round(sim, 4),
                    "confidence": round(conf_c, 4),
                    "conf_change": round(conf_change, 4),
                }

        status = f"iter {i+1}/{max_iters} | best conf change so far: {best['conf_change'] if best else 'none'}"
        print(f"  {status}")

        if best and best["conf_change"] >= min_conf_change:
            print(f"  [OK] Strong example found — stopping early.")
            break

    if best is None:
        return None
    if best["conf_change"] < min_conf_change:
        print(f"  [WARN] Best confidence change {best['conf_change']} is below threshold {min_conf_change} — discarding.")
        return None

    return best


def print_result(source, best):
    original_label_name = "deceptive" if source["original_label"] == 1 else "truthful"
    target_label_name = "truthful" if source["original_label"] == 1 else "deceptive"

    print(f"\n{'='*70}")
    print(f"Direction: {source['direction']}")
    print(f"Original text (classified as {original_label_name}):")
    print(textwrap.fill(source["text"], width=80, initial_indent="  ", subsequent_indent="  "))
    print()

    if best is None:
        print("  [No strong paraphrase found — try more iterations, candidates, or a different model]")
        return

    print(f"  Best paraphrase: similarity={best['similarity']} | conf change={best['conf_change']} | confidence={best['confidence']} ({target_label_name})")
    print(textwrap.fill(best["text"], width=80, initial_indent="      ", subsequent_indent="      "))
    print()
    print("  --- Copy-paste format for paraphraser.py ---")
    print(f"  Original (classified as {original_label_name}): \"{source['text']}\"")
    print(f"  Rewritten (to be classified as {target_label_name}):")
    print(f"  {best['text']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate and validate example paraphrases")
    parser.add_argument("--model", type=str, required=True,
                        help="Ollama model to use as the neutral generator (e.g. phi4, llama3.1:8b)")
    parser.add_argument("--candidates", type=int, default=10,
                        help="Number of paraphrase candidates to generate per source text (default: 10)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Max iterations per source text (default: 10)")
    parser.add_argument("--min-conf-change", type=float, default=0.3,
                        help="Minimum confidence change to accept a paraphrase as a strong example (default: 0.3)")
    args = parser.parse_args()

    print(f"Loading classifier...")
    classifier = DeceptionClassifier()

    print(f"Using model: {args.model} | candidates: {args.candidates} | iters: {args.iters} | min conf change: {args.min_conf_change}")
    paraphraser = Paraphraser(model=args.model, strategy="zero_shot")

    for source in SOURCE_TEXTS:
        print(f"\nGenerating for: {source['direction']} ...")
        best = find_best_paraphrase(source, classifier, paraphraser,
                                    n_candidates=args.candidates,
                                    max_iters=args.iters,
                                    min_conf_change=args.min_conf_change)
        print_result(source, best)


if __name__ == "__main__":
    main()
