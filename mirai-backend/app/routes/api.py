from fastapi import APIRouter

from app.models.schemas import (
    AnalyticsRequest,
    AnalyticsResponse,
    ContentResponse,
    EmotionResponse,
    GenerateRequest,
    GenerateResponse,
    SentimentRequest,
    TrendsResponse,
)
from app.services.content_service import content_service
from app.services.emotion_service import emotion_service
from app.services.engagement_service import engagement_service
from app.services.rag_service import rag_service
from app.services.trends_service import trends_service

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    emotion_result = emotion_service.detect(request.text)
    rag_result = rag_service.retrieve(query=f"{request.topic}. {request.text}", top_k=3)

    content_result = content_service.generate(
        topic=request.topic,
        emotions=emotion_result.emotions,
        context=rag_result.context,
    )
    engagement_result = engagement_service.predict_from_emotions(
        emotions=emotion_result.emotions,
        scores=emotion_result.scores,
    )

    return {
        "emotions": emotion_result.emotions,
        "context": rag_result.context,
        "content": ContentResponse(
            caption=content_result.caption,
            hashtags=content_result.hashtags,
            tone=content_result.tone,
        ),
        "engagement": {
            "score": engagement_result.score,
            "label": engagement_result.label,
        },
    }


@router.post("/sentiment", response_model=EmotionResponse)
def sentiment(request: SentimentRequest):
    result = emotion_service.detect(request.text)
    return {"emotions": result.emotions, "scores": result.scores}


@router.get("/trends", response_model=TrendsResponse)
def trends():
    return {"trends": trends_service.get_trends()}


@router.post("/analytics", response_model=AnalyticsResponse)
def analytics(request: AnalyticsRequest):
    return {"engagement_score": engagement_service.analytics_score(request.likes, request.comments)}
