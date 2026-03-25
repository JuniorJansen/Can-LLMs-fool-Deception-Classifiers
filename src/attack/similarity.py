from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def compute_similarity(original, candidate):
    model = _get_model()
    embeddings = model.encode([original, candidate])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

def compute_similarities_batch(original_embedding, candidates):
    """Compute similarities for multiple candidates against a pre-computed original embedding."""
    if not candidates:
        return []
    model = _get_model()
    candidate_embeddings = model.encode(candidates)
    scores = cosine_similarity([original_embedding], candidate_embeddings)[0]
    return [float(s) for s in scores]

def encode_text(text):
    """Encode a single text into an embedding vector."""
    model = _get_model()
    return model.encode(text)

def is_similar_enough(original, candidate, threshold):
    return compute_similarity(original, candidate) >= threshold
