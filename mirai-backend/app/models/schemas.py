from typing import List

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=3000)


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)


class AnalyticsRequest(BaseModel):
    likes: int = Field(..., ge=0)
    comments: int = Field(..., ge=0)


class EmotionResponse(BaseModel):
    emotions: List[str]
    scores: List[float]


class ContextResponse(BaseModel):
    context: List[str]


class ContentResponse(BaseModel):
    caption: str
    hashtags: str
    tone: str


class EngagementResponse(BaseModel):
    engagement_score: float
    engagement_label: str


class GenerateResponse(BaseModel):
    emotions: List[str]
    context: List[str]
    content: ContentResponse
    engagement: dict


class TrendsResponse(BaseModel):
    trends: List[str]


class AnalyticsResponse(BaseModel):
    engagement_score: float
