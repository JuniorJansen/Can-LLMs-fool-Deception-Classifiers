"""Shared paths, hyperparameters, and reproducibility settings."""

import os
import random

import numpy as np
import torch

# Paths
MODEL_PATH = "models/DistilBERT"
DATA_PATH = "data/hippocorpus_test_truncated.csv"

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Attack hyperparameters (matched to the thesis)
MAX_ITER = 10
NUM_CANDIDATES = 5
SIMILARITY_THRESHOLD = 0.85
LENGTH_TOLERANCE = 0.10

# Reproducibility
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set every RNG we rely on; called once at program start."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
