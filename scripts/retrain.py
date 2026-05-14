"""Re-train a logistic-regression head on mean-pooled DistilBERT embeddings.

Usage:
    python scripts/retrain.py --train data/train.csv --out models/candidate.pkl
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from app.config import MODEL_NAME


def embed(texts, tokenizer, model, batch_size: int = 16) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            h = model(**enc).last_hidden_state  # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    assert {"text", "label"} <= set(df.columns), "train.csv must have text,label"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = AutoModel.from_pretrained(MODEL_NAME)

    X = embed(df["text"].astype(str).tolist(), tok, enc)
    y = df["label"].astype(int).values

    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(X, y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "encoder": MODEL_NAME}, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
