"""Sentence-BERT cosine similarity used by the attack filter.

The encoder (all-MiniLM-L6-v2) is loaded lazily on first use so that scripts
that don't need it (e.g. analyze_stats) don't pay the import cost.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def encode_text(text):
    """Encode a single text into a fixed-size embedding vector."""
    return _get_model().encode(text)


def compute_similarity(original, candidate):
    """Cosine similarity between two texts (re-encodes both)."""
    embeddings = _get_model().encode([original, candidate])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


def compute_similarities_batch(original_embedding, candidates):
    """Cosine similarity of a list of candidates against a pre-computed reference embedding.

    Used inside the attack loop, where the reference (original narrative) is
    encoded once per attack and then re-used for every candidate.
    """
    if not candidates:
        return []
    candidate_embeddings = _get_model().encode(candidates)
    scores = cosine_similarity([original_embedding], candidate_embeddings)[0]
    return [float(s) for s in scores]
