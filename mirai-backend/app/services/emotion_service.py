from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
THRESHOLD = 0.3


@dataclass
class EmotionResult:
    emotions: List[str]
    scores: List[float]


class EmotionService:
    """Multi-label emotion detection using a pretrained GoEmotions model."""

    def __init__(self, model_name: str = MODEL_NAME, threshold: float = THRESHOLD) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._tokenizer = None
        self._model = None
        self._lock = Lock()

    def _lazy_load(self) -> None:
        if self._model is None or self._tokenizer is None:
            with self._lock:
                if self._model is None or self._tokenizer is None:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                    self._model.eval()

    def detect(self, text: str) -> EmotionResult:
        self._lazy_load()

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self._model(**encoded).logits
            probabilities = torch.sigmoid(logits).squeeze(0)

        id2label: Dict[int, str] = self._model.config.id2label
        selected = [
            (id2label[idx], float(score))
            for idx, score in enumerate(probabilities.tolist())
            if score >= self.threshold
        ]

        if not selected:
            max_idx = int(torch.argmax(probabilities).item())
            selected = [(id2label[max_idx], float(probabilities[max_idx].item()))]

        selected.sort(key=lambda item: item[1], reverse=True)
        emotions = [label for label, _ in selected]
        scores = [round(score, 4) for _, score in selected]

        return EmotionResult(emotions=emotions, scores=scores)


emotion_service = EmotionService()
