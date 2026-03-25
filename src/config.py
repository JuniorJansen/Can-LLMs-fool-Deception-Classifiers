import torch
MODEL_PATH = "models/DistilBERT"
DATA_PATH = "data/hippocorpus_test_truncated.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ITER = 10
NUM_CANDIDATES = 5
SIMILARITY_THRESHOLD = 0.75
