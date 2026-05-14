"""HuggingFace DistilBERT-SST2 wrapper."""
import time
from typing import Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MODEL_NAME, MODEL_VERSION


class SentimentModel:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.version = MODEL_VERSION
        self.loaded = False

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.loaded = True

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float, float]:
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        label = self.model.config.id2label[idx]
        score = float(probs[idx])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return label, score, latency_ms


_model: SentimentModel | None = None


def get_model() -> SentimentModel:
    global _model
    if _model is None:
        _model = SentimentModel()
        _model.load()
    return _model
