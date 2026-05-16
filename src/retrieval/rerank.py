"""Reranker. Cohere Rerank v3 in prod (paid), cross-encoder local fallback."""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

from loguru import logger

from src.api.schemas import RetrievedProduct
from src.config import get_settings


class Reranker(Protocol):
    async def rerank(self, query: str, candidates: Sequence[RetrievedProduct], top_n: int) -> list[RetrievedProduct]: ...


# ---------------------------------------------------------------------------
# COHERE (paid, recommended)
# ---------------------------------------------------------------------------

class CohereReranker:
    def __init__(self, api_key: str) -> None:
        import cohere
        self.client = cohere.AsyncClient(api_key=api_key)
        self.model = "rerank-english-v3.0"

    async def rerank(self, query: str, candidates: Sequence[RetrievedProduct], top_n: int) -> list[RetrievedProduct]:
        if not candidates:
            return []
        docs = [f"{c.product_name} ({c.aisle} / {c.department})" for c in candidates]
        result = await self.client.rerank(model=self.model, query=query, documents=docs, top_n=top_n)
        out: list[RetrievedProduct] = []
        for r in result.results:
            c = candidates[r.index]
            out.append(RetrievedProduct(**{**c.model_dump(), "score": float(r.relevance_score), "source": "rerank"}))
        return out


# ---------------------------------------------------------------------------
# LOCAL cross-encoder fallback (free)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading local cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    async def rerank(self, query: str, candidates: Sequence[RetrievedProduct], top_n: int) -> list[RetrievedProduct]:
        if not candidates:
            return []
        pairs = [(query, f"{c.product_name} ({c.aisle} / {c.department})") for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False).tolist()
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            RetrievedProduct(**{**c.model_dump(), "score": float(s), "source": "rerank"})
            for c, s in ranked
        ]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    s = get_settings()
    if s.rerank_backend == "cohere" and s.cohere_api_key:
        logger.info("Using Cohere Rerank v3 (paid)")
        return CohereReranker(s.cohere_api_key)
    logger.info("Using local cross-encoder reranker (free)")
    return CrossEncoderReranker()
