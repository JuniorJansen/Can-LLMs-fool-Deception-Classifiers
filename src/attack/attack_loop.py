"""Iterative black-box paraphrase attack loop.

For each input narrative, up to ``max_iter`` rounds are run. Each round asks
the LLM for ``k`` candidate paraphrases, filters them by length and Sentence-
BERT similarity, and scores the survivors with the target classifier. The
candidate that produces the largest drop in the *original* label's probability
is the iteration's best; if it flips the prediction, the attack succeeds.

All candidates are generated from (and compared against) the original
narrative, never from the previous iteration's best, which prevents semantic
drift across iterations.
"""

from src.attack.similarity import (
    compute_similarities_batch,
    compute_similarity,
    encode_text,
)
from src.config import LENGTH_TOLERANCE, MAX_ITER, NUM_CANDIDATES, SIMILARITY_THRESHOLD


def _filter_by_length(candidates, reference_text, tolerance=LENGTH_TOLERANCE):
    """Discard candidates whose word count deviates from the reference by more than ``tolerance``."""
    ref_len = len(reference_text.split())
    lo = ref_len * (1 - tolerance)
    hi = ref_len * (1 + tolerance)
    return [c for c in candidates if lo <= len(c.split()) <= hi]


def run_attack(text, classifier, paraphraser, max_iter=MAX_ITER, k=NUM_CANDIDATES):
    original_label, original_conf = classifier.predict(text)
    original_embedding = encode_text(text)

    best_text = text
    best_conf = original_conf
    max_conf_change = 0
    iteration_log = []

    for iteration in range(max_iter):
        # Try up to twice to get k unique, length-valid, similarity-valid candidates.
        candidates = []
        for _ in range(2):
            if len(candidates) >= k:
                break
            raw = paraphraser.generate(text, k=k, original_label=original_label)
            raw = [c for c in raw if c != text and c not in [x[0] for x in candidates]]
            raw = _filter_by_length(raw, text)
            if raw:
                sims = compute_similarities_batch(original_embedding, raw)
                candidates.extend((c, sim) for c, sim in zip(raw, sims) if sim >= SIMILARITY_THRESHOLD)
        candidates = candidates[:k]

        # Pick the candidate that pushed the original label's probability down the most.
        # Tracking the original-label probability (rather than the predicted-label confidence)
        # keeps successful and unsuccessful candidates on the same scale.
        iter_best_text = best_text
        iter_best_change = 0
        iter_best_label = original_label
        iter_best_conf = best_conf
        iter_best_sim = None

        for candidate, sim in candidates:
            pred, conf = classifier.predict(candidate)
            original_label_prob = conf if pred == original_label else 1 - conf
            change = original_conf - original_label_prob
            if change > iter_best_change:
                iter_best_change = change
                iter_best_text = candidate
                iter_best_label = pred
                iter_best_conf = conf
                iter_best_sim = sim

        if iter_best_change > max_conf_change:
            max_conf_change = iter_best_change
            best_text = iter_best_text
            best_conf = iter_best_conf

        iteration_log.append({
            "iteration": iteration + 1,
            "num_candidates": len(candidates),
            "best_confidence": round(iter_best_conf, 4),
            "confidence_change": round(iter_best_change, 4),
            "similarity": round(iter_best_sim, 4) if iter_best_sim is not None else None,
        })

        if iter_best_label != original_label:
            return _result(True, iter_best_text, original_label, iter_best_label,
                           original_conf, iter_best_conf, iteration + 1,
                           iter_best_change, max_conf_change,
                           compute_similarity(text, iter_best_text), iteration_log)

    return _result(False, best_text, original_label, iter_best_label,
                   original_conf, best_conf, max_iter,
                   max_conf_change, max_conf_change,
                   compute_similarity(text, best_text), iteration_log)


def _result(success, final_text, original_label, new_label, original_conf, final_conf,
            iterations, confidence_change, max_confidence_change, final_similarity, log):
    return {
        "success": success,
        "final_text": final_text,
        "original_label": original_label,
        "new_label": new_label,
        "original_confidence": round(original_conf, 4),
        "final_confidence": round(final_conf, 4),
        "iterations": iterations,
        "confidence_change": round(confidence_change, 4),
        "max_confidence_change": round(max_confidence_change, 4),
        "final_similarity": round(final_similarity, 4),
        "iteration_log": log,
    }
