import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    print("Columns:", df.columns)
    print(df.head())
    print("Unique labels:", df["condition"].unique())

    texts = df["text_truncated"].tolist()

    label_map = {
        "truthful": 0,
        "deceptive": 1
    }

    labels = [label_map[l] for l in df["condition"]]

    return texts, labels