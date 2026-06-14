"""HippoCorpus loader.

Reads the preprocessed test split (with the ``text_truncated`` column already
fitted to DistilBERT's 512-token budget) and returns parallel lists of texts
and binary labels: 0 = truthful (recalled), 1 = deceptive (imagined).
"""

import pandas as pd

LABEL_MAP = {"truthful": 0, "deceptive": 1}


def load_dataset(path):
    df = pd.read_csv(path)
    texts = df["text_truncated"].tolist()
    labels = [LABEL_MAP[c] for c in df["condition"]]
    return texts, labels
