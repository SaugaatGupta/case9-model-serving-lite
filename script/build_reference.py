"""Build models/reference_stats.json from 1000 SST-2 dev samples."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from app.config import REFERENCE_N, REFERENCE_STATS_PATH
from app.drift import build_reference, save_reference


def main(n: int) -> None:
    ds = load_dataset("sst2", split="validation")
    take = min(n, len(ds))
    texts = [ds[i]["sentence"] for i in range(take)]
    labels = ["POSITIVE" if ds[i]["label"] == 1 else "NEGATIVE" for i in range(take)]
    stats_obj = build_reference(texts, labels)
    save_reference(stats_obj)
    print(f"wrote {REFERENCE_STATS_PATH} (n={stats_obj['n']})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=REFERENCE_N)
    main(p.parse_args().n)
