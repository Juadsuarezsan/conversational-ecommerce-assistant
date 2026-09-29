"""Rerankers behind one :class:`Reranker` protocol.

* ``cohere`` — Cohere Rerank v3 (paid, production).
* ``local`` — ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (free, needs the ``ml`` extra;
  imported lazily).
* ``lexical`` — deterministic token-overlap scorer. No model, no network. It is
  the offline default and is reported as *deterministic lexical fallback*.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import RetrievedProduct
from src.config import get_settings
from src.retrieval.bm25 import tokenize


class Reranker(Protocol):
    """Re-orders retrieval candidates for a query and truncates to ``top_n``."""

    name: str

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedProduct], top_n: int
    ) -> list[RetrievedProduct]:
        """Return the ``top_n`` candidates most relevant to ``query``."""
        ...


def _doc_text(c: RetrievedProduct) -> str:
    return f"{c.product_name} ({c.aisle} / {c.department})"


def _with_score(c: RetrievedProduct, score: float) -> RetrievedProduct:
    return RetrievedProduct(**{**c.model_dump(), "score": float(score), "source": "rerank"})


# ---------------------------------------------------------------------------
# Deterministic lexical fallback (offline default)
# ---------------------------------------------------------------------------


class LexicalReranker:
    """Token-overlap reranker: fraction of query tokens present in the product text.

    Ties are broken by the candidate's incoming score, so the fused (RRF) order is
    preserved when the lexical signal is uninformative. Used when neither Cohere
    nor the local cross-encoder is available.
    """

    name = "lexical"

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedProduct], top_n: int
    ) -> list[RetrievedProduct]:
        """Rerank by query-token coverage, then by incoming score."""
        if not candidates:
            return []
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return [_with_score(c, c.score) for c in candidates[:top_n]]
        scored: list[tuple[float, float, RetrievedProduct]] = []
        for c in candidates:
            d_tokens = set(tokenize(_doc_text(c)))
            overlap = len(q_tokens & d_tokens) / len(q_tokens)
            name_bonus = 0.25 if q_tokens & set(tokenize(c.product_name)) else 0.0
            scored.append((overlap + name_bonus, c.score, c))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [_with_score(c, s) for s, _, c in scored[:top_n]]


# ---------------------------------------------------------------------------
# Cohere (paid)
# ---------------------------------------------------------------------------


class CohereReranker:
    """Cohere ``rerank-english-v3.0`` with retries and an explicit timeout."""

    name = "cohere-rerank-v3"

    def __init__(
        self, api_key: str, model: str = "rerank-english-v3.0", timeout: float = 10.0
    ) -> None:
        import cohere

        self.client = cohere.AsyncClient(api_key=api_key, timeout=timeout)
        self.model = model
        self.name = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def rerank(
        self, query: str, candidates: Sequence[RetrievedProduct], top_n: int
    ) -> list[RetrievedProduct]:
        """Call Cohere Rerank and map results back to the candidates."""
        if not candidates:
            return []
        docs = [_doc_text(c) for c in candidates]
        result = await self.client.rerank(
            model=self.model, query=query, documents=docs, top_n=top_n
        )
        return [_with_score(candidates[r.index], r.relevance_score) for r in result.results]


# ---------------------------------------------------------------------------
# Local cross-encoder (free, optional ``ml`` extra)
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """``ms-marco-MiniLM-L-6-v2`` cross-encoder; ``sentence_transformers`` imported lazily."""

    name = "ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "RERANK_BACKEND=local requires the 'ml' extra: pip install -e '.[ml]'"
            ) from exc
        logger.info("Loading local cross-encoder: {}", model_name)
        self.model = CrossEncoder(model_name)
        self.name = model_name.rsplit("/", 1)[-1]

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedProduct], top_n: int
    ) -> list[RetrievedProduct]:
        """Score (query, product) pairs with the cross-encoder."""
        if not candidates:
            return []
        pairs = [(query, _doc_text(c)) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False).tolist()
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [_with_score(c, s) for c, s in ranked[:top_n]]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Build the reranker selected by ``RERANK_BACKEND`` (cached per process)."""
    s = get_settings()
    if s.rerank_backend == "cohere":
        if not s.cohere_api_key:
            raise RuntimeError("RERANK_BACKEND=cohere requires COHERE_API_KEY")
        logger.info("Reranker: Cohere {}", s.cohere_rerank_model)
        return CohereReranker(
            s.cohere_api_key, model=s.cohere_rerank_model, timeout=s.external_timeout_seconds
        )
    if s.rerank_backend == "local":
        logger.info("Reranker: local cross-encoder")
        return CrossEncoderReranker(s.local_rerank_model)
    logger.warning("Reranker: deterministic lexical fallback (no model).")
    return LexicalReranker()
