from src.attack.similarity import encode_text, compute_similarities_batch, compute_similarity
from src.config import SIMILARITY_THRESHOLD, MAX_ITER, NUM_CANDIDATES


def run_attack(text, classifier, paraphraser, max_iter=MAX_ITER, k=NUM_CANDIDATES):
    original_label, original_conf = classifier.predict(text)
    original_embedding = encode_text(text)

    current_text = text
    current_conf = original_conf

    max_conf_change = 0
    iteration_log = []

    for iteration in range(max_iter):
        candidates = []
        attempts = 0
        max_attempts = 2
        while len(candidates) < k and attempts < max_attempts:
            raw = paraphraser.generate(current_text, k=k, original_label=original_label)
            raw = [c for c in raw if c not in [x[0] for x in candidates]]
            if raw:
                sims = compute_similarities_batch(original_embedding, raw)
                for c, sim in zip(raw, sims):
                    if sim >= SIMILARITY_THRESHOLD:
                        candidates.append((c, sim))
            attempts += 1
        candidates = candidates[:k]

        best_text = current_text
        best_change = 0
        best_label = original_label
        best_conf = current_conf
        best_sim = None

        for candidate, sim in candidates:
            pred, conf = classifier.predict(candidate)
            change = abs(conf - original_conf)

            if change > best_change:
                best_change = change
                best_text = candidate
                best_label = pred
                best_conf = conf
                best_sim = sim

        current_text = best_text
        current_conf = best_conf
        max_conf_change = max(max_conf_change, best_change)

        iteration_log.append({
            "iteration": iteration + 1,
            "num_candidates": len(candidates),
            "best_confidence": round(best_conf, 4),
            "confidence_change": round(best_change, 4),
            "similarity": round(best_sim, 4) if best_sim is not None else None,
        })

        if best_label != original_label:
            final_sim = compute_similarity(text, current_text)
            return {
                "success": True,
                "final_text": current_text,
                "original_label": original_label,
                "new_label": best_label,
                "original_confidence": round(original_conf, 4),
                "final_confidence": round(best_conf, 4),
                "iterations": iteration + 1,
                "confidence_change": round(best_change, 4),
                "max_confidence_change": round(max_conf_change, 4),
                "final_similarity": round(final_sim, 4),
                "iteration_log": iteration_log,
            }

    final_sim = compute_similarity(text, current_text)
    return {
        "success": False,
        "final_text": current_text,
        "original_label": original_label,
        "new_label": best_label,
        "original_confidence": round(original_conf, 4),
        "final_confidence": round(best_conf, 4),
        "iterations": max_iter,
        "confidence_change": round(best_change, 4),
        "max_confidence_change": round(max_conf_change, 4),
        "final_similarity": round(final_sim, 4),
        "iteration_log": iteration_log,
    }
