from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.api import router

app = FastAPI(
    title="mirAI Backend",
    version="1.0.0",
    description="AI-powered backend for emotion-aware social media content generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(router)
