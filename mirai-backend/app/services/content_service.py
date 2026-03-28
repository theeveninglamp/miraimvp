from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import google.generativeai as genai


@dataclass
class ContentResult:
    caption: str
    hashtags: str
    tone: str


class ContentService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._model = None

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

    def _tone_from_emotions(self, emotions: List[str]) -> str:
        emotion_set = set(emotions)
        if any(e in emotion_set for e in ["joy", "amusement", "excitement", "optimism"]):
            return "engaging"
        if any(e in emotion_set for e in ["sadness", "grief", "disappointment"]):
            return "emotional"
        return "informative"

    def _template_fallback(self, topic: str, emotions: List[str], context: List[str]) -> ContentResult:
        tone = self._tone_from_emotions(emotions)
        top_emotions = ", ".join(emotions[:3]) if emotions else "focused"
        key_tip = context[0] if context else "Give clear value and stay consistent with your posting routine."

        caption = (
            f"{topic.title()} can feel intense, but you're not alone. "
            f"Today I'm channeling {top_emotions} into growth. {key_tip} "
            "If this resonates, drop your biggest challenge below and let's solve it together."
        )

        hashtags = [
            f"#{topic.strip().replace(' ', '')}",
            "#ContentCreator",
            "#SocialMediaGrowth",
            "#ReelsTips",
            "#CreatorJourney",
        ]
        return ContentResult(caption=caption, hashtags=" ".join(hashtags), tone=tone)

    def generate(self, topic: str, emotions: List[str], context: List[str]) -> ContentResult:
        if self._model is None:
            return self._template_fallback(topic, emotions, context)

        prompt = (
            "You are a social media strategist. Create one short high-performing caption and hashtags. "
            f"Topic: {topic}. Detected emotions: {', '.join(emotions)}. "
            f"Use these context tips: {' | '.join(context)}. "
            "Return JSON with keys caption, hashtags, tone only."
        )

        try:
            response = self._model.generate_content(prompt)
            text = (response.text or "").strip()
            # Best-effort parse: if JSON format is not guaranteed, fallback.
            if '"caption"' in text and '"hashtags"' in text and '"tone"' in text:
                # Lightweight extraction to avoid adding extra dependencies.
                import json
                import re

                match = re.search(r"\{[\s\S]*\}", text)
                if match:
                    payload = json.loads(match.group(0))
                    return ContentResult(
                        caption=str(payload.get("caption", "")).strip() or self._template_fallback(topic, emotions, context).caption,
                        hashtags=str(payload.get("hashtags", "")).strip() or "#ContentCreator #SocialMediaGrowth",
                        tone=str(payload.get("tone", "")).strip() or self._tone_from_emotions(emotions),
                    )
        except Exception:
            pass

        return self._template_fallback(topic, emotions, context)


content_service = ContentService()
