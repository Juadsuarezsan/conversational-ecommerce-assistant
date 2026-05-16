"""Embedding providers. Voyage AI in prod (paid), sentence-transformers fallback (free local).

Both produce 1024-dim vectors so the same Qdrant/pgvector schema works either way.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from loguru import logger

from src.config import get_settings


class EmbeddingProvider(Protocol):
    dim: int
    async def embed_query(self, text: str) -> list[float]: ...
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Voyage AI (paid, recommended for production)
# ---------------------------------------------------------------------------

class VoyageEmbeddings:
    """voyage-3 is 1024-dim natively."""

    dim = 1024

    def __init__(self, api_key: str) -> None:
        import voyageai
        self.client = voyageai.AsyncClient(api_key=api_key)
        self.model = "voyage-3"

    async def embed_query(self, text: str) -> list[float]:
        result = await self.client.embed([text], model=self.model, input_type="query")
        return result.embeddings[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = await self.client.embed(texts, model=self.model, input_type="document")
        return result.embeddings


# ---------------------------------------------------------------------------
# Local fallback — sentence-transformers all-MiniLM-L6-v2 (384-dim, padded)
# ---------------------------------------------------------------------------

class LocalEmbeddings:
    """Free local fallback. all-MiniLM is 384-dim; we pad/project to 1024 to match the schema."""

    dim = 1024

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._native_dim = self.model.get_sentence_embedding_dimension()

    def _pad(self, vec: list[float]) -> list[float]:
        if len(vec) >= self.dim:
            return vec[: self.dim]
        return vec + [0.0] * (self.dim - len(vec))

    async def embed_query(self, text: str) -> list[float]:
        emb = self.model.encode(text, normalize_embeddings=True).tolist()
        return self._pad(emb)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        return [self._pad(e) for e in embs]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    s = get_settings()
    if s.embedding_backend == "voyage" and s.voyage_api_key:
        logger.info("Using Voyage AI embeddings (voyage-3)")
        return VoyageEmbeddings(s.voyage_api_key)
    logger.info("Using local sentence-transformers embeddings (free)")
    return LocalEmbeddings()
