from src.attack.similarity import encode_text, compute_similarities_batch, compute_similarity
from src.config import SIMILARITY_THRESHOLD, MAX_ITER, NUM_CANDIDATES

LENGTH_TOLERANCE = 0.1


def _filter_by_length(candidates, reference_text, tolerance=LENGTH_TOLERANCE):
    """Reject candidates whose word count deviates more than tolerance from the reference."""
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
        candidates = []
        attempts = 0
        max_attempts = 2
        while len(candidates) < k and attempts < max_attempts:
            raw = paraphraser.generate(text, k=k, original_label=original_label)
            raw = [c for c in raw if c != text and c not in [x[0] for x in candidates]]
            raw = _filter_by_length(raw, text)
            if raw:
                sims = compute_similarities_batch(original_embedding, raw)
                for c, sim in zip(raw, sims):
                    if sim >= SIMILARITY_THRESHOLD:
                        candidates.append((c, sim))
            attempts += 1
        candidates = candidates[:k]

        iter_best_text = best_text
        iter_best_change = 0
        iter_best_label = original_label
        iter_best_conf = best_conf
        iter_best_sim = None

        for candidate, sim in candidates:
            pred, conf = classifier.predict(candidate)
            # Always measure how much the original label's probability dropped,
            # so failed and successful attacks are on the same scale.
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
            final_sim = compute_similarity(text, iter_best_text)
            return {
                "success": True,
                "final_text": iter_best_text,
                "original_label": original_label,
                "new_label": iter_best_label,
                "original_confidence": round(original_conf, 4),
                "final_confidence": round(iter_best_conf, 4),
                "iterations": iteration + 1,
                "confidence_change": round(iter_best_change, 4),
                "max_confidence_change": round(max_conf_change, 4),
                "final_similarity": round(final_sim, 4),
                "iteration_log": iteration_log,
            }

    final_sim = compute_similarity(text, best_text)
    return {
        "success": False,
        "final_text": best_text,
        "original_label": original_label,
        "new_label": iter_best_label,
        "original_confidence": round(original_conf, 4),
        "final_confidence": round(best_conf, 4),
        "iterations": max_iter,
        "confidence_change": round(max_conf_change, 4),
        "max_confidence_change": round(max_conf_change, 4),
        "final_similarity": round(final_sim, 4),
        "iteration_log": iteration_log,
    }
