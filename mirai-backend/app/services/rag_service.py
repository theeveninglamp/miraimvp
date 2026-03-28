from __future__ import annotations

from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


KNOWLEDGE_BASE: List[str] = [
    "Use a strong hook in the first three seconds of your reel to stop users from scrolling.",
    "Post consistently at the same time daily so your audience and algorithm both learn your schedule.",
    "Share one personal story each week to build trust and emotional connection.",
    "Convert comments into content ideas by answering frequent questions in short videos.",
    "Use captions on videos because many viewers watch without sound in public.",
    "Pair trending audio with niche advice to improve discoverability and relevance.",
    "End each post with a clear call-to-action like ask, save, or share to boost engagement.",
    "Carousel posts with actionable tips increase saves and improve long-term distribution.",
    "Track watch time and retention rate to identify which storytelling formats keep people engaged.",
    "Collaborate with micro-creators in your niche to cross-pollinate audiences with high trust.",
    "Use before-and-after style transformations to make progress visible and memorable.",
    "Reply to early comments in the first hour to improve momentum and community signals.",
]


@dataclass
class RetrievalResult:
    context: List[str]


class RagService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = KNOWLEDGE_BASE
        self.embeddings = self.model.encode(self.documents, convert_to_numpy=True, normalize_embeddings=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        _, indices = self.index.search(query_embedding, top_k)
        selected = [self.documents[i] for i in indices[0] if 0 <= i < len(self.documents)]
        return RetrievalResult(context=selected)


rag_service = RagService()
