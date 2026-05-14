"""Evaluate a candidate model against eval/held_out.csv.

Usage:
    python scripts/evaluate.py --model models/candidate.pkl --eval eval/held_out.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer

from scripts.retrain import embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["clf"]
    tok = AutoTokenizer.from_pretrained(bundle["encoder"])
    enc = AutoModel.from_pretrained(bundle["encoder"])

    df = pd.read_csv(args.eval)
    X = embed(df["text"].astype(str).tolist(), tok, enc)
    y_true = df["label"].astype(int).values
    y_pred = clf.predict(X)

    metrics = {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
