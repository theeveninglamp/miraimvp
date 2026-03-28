from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EngagementResult:
    score: float
    label: str


class EngagementService:
    emotion_weights: Dict[str, float] = {
        "joy": 0.9,
        "excitement": 0.92,
        "amusement": 0.95,
        "sadness": 0.65,
        "anger": 0.55,
        "admiration": 0.8,
        "optimism": 0.85,
        "curiosity": 0.78,
        "surprise": 0.82,
        "nervousness": 0.72,
        "fear": 0.58,
    }

    def predict_from_emotions(self, emotions: List[str], scores: List[float]) -> EngagementResult:
        if not emotions or not scores:
            return EngagementResult(score=50.0, label="Medium")

        weighted_sum = 0.0
        score_sum = 0.0
        for emotion, confidence in zip(emotions, scores):
            weight = self.emotion_weights.get(emotion, 0.7)
            weighted_sum += weight * confidence
            score_sum += confidence

        normalized = (weighted_sum / score_sum) if score_sum > 0 else 0.7
        engagement_score = round(min(max(normalized * 100, 0), 100), 2)

        if engagement_score >= 75:
            label = "High"
        elif engagement_score >= 50:
            label = "Medium"
        else:
            label = "Low"

        return EngagementResult(score=engagement_score, label=label)

    def analytics_score(self, likes: int, comments: int) -> float:
        score = min(100.0, round((likes * 0.4 + comments * 1.2), 2))
        return score


engagement_service = EngagementService()
