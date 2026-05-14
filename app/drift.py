"""Drift signals: length KS-test, non-English ratio, vocab Jaccard, label shift."""
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

from scipy import stats

from .config import (
    DRIFT_JACCARD_MIN,
    DRIFT_LABEL_DELTA,
    DRIFT_LENGTH_KS_P,
    DRIFT_NON_ENGLISH_RATIO,
    REFERENCE_STATS_PATH,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def build_reference(texts: List[str], labels: List[str]) -> dict:
    lengths = [len(t) for t in texts]
    tokens: Counter = Counter()
    for t in texts:
        tokens.update(tokenize(t))
    top_100 = [w for w, _ in tokens.most_common(100)]
    pos = sum(1 for label in labels if str(label).upper() == "POSITIVE")
    total = max(len(labels), 1)
    return {
        "n": len(texts),
        "lengths": lengths,
        "top_100_tokens": top_100,
        "label_positive_ratio": pos / total,
    }


def load_reference(path: Optional[Path] = None) -> Optional[dict]:
    p = Path(path or REFERENCE_STATS_PATH)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_reference(stats_obj: dict, path: Optional[Path] = None) -> None:
    p = Path(path or REFERENCE_STATS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats_obj))


class DriftState:
    """In-memory rolling cache of tokenised text — needed for Jaccard since raw
    text is not persisted to the event store (privacy)."""

    def __init__(self, max_items: int = 500) -> None:
        self.max_items = max_items
        self.token_buffer: list[set] = []

    def add(self, text: str) -> None:
        toks = set(tokenize(text))
        self.token_buffer.append(toks)
        if len(self.token_buffer) > self.max_items:
            self.token_buffer = self.token_buffer[-self.max_items :]

    def union_tokens(self) -> set:
        u: set = set()
        for s in self.token_buffer:
            u |= s
        return u

    def jaccard_vs(self, top_tokens: List[str]) -> float:
        ref = set(top_tokens)
        win = self.union_tokens()
        if not ref and not win:
            return 1.0
        union = len(ref | win)
        if not union:
            return 1.0
        return len(ref & win) / union


drift_state = DriftState()


def compute_drift(window: List[dict], reference: dict) -> dict:
    if not window or not reference:
        return {
            "overall_status": "insufficient_data",
            "window_size": len(window),
            "signals": [],
        }

    # (a) length KS test
    win_lengths = [e["text_len"] for e in window]
    ref_lengths = reference.get("lengths", [])
    if ref_lengths and win_lengths:
        try:
            ks_p = float(stats.ks_2samp(win_lengths, ref_lengths).pvalue)
        except Exception:
            ks_p = 1.0
    else:
        ks_p = 1.0
    length_triggered = ks_p < DRIFT_LENGTH_KS_P

    # (b) non-English ratio
    n = len(window)
    non_en = sum(1 for e in window if (e.get("text_lang") or "en") != "en")
    non_en_ratio = non_en / n if n else 0.0
    lang_triggered = non_en_ratio > DRIFT_NON_ENGLISH_RATIO

    # (c) top-100 vocabulary Jaccard
    jaccard_val = drift_state.jaccard_vs(reference.get("top_100_tokens", []))
    jaccard_triggered = jaccard_val < DRIFT_JACCARD_MIN

    # (d) label distribution shift
    pos = sum(1 for e in window if str(e["label"]).upper() == "POSITIVE")
    pos_ratio = pos / n if n else 0.0
    label_delta = abs(pos_ratio - reference.get("label_positive_ratio", 0.5))
    label_triggered = label_delta > DRIFT_LABEL_DELTA

    signals = [
        {"name": "length_ks_pvalue", "value": ks_p,
         "threshold": DRIFT_LENGTH_KS_P, "triggered": length_triggered},
        {"name": "non_english_ratio", "value": non_en_ratio,
         "threshold": DRIFT_NON_ENGLISH_RATIO, "triggered": lang_triggered},
        {"name": "vocab_jaccard", "value": jaccard_val,
         "threshold": DRIFT_JACCARD_MIN, "triggered": jaccard_triggered},
        {"name": "label_positive_delta", "value": label_delta,
         "threshold": DRIFT_LABEL_DELTA, "triggered": label_triggered},
    ]
    return {
        "overall_status": "drift_detected" if any(s["triggered"] for s in signals) else "ok",
        "window_size": n,
        "signals": signals,
    }
