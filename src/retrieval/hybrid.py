"""Hybrid search: BM25 + dense vectors fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from src.api.schemas import RetrievedProduct
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseStore
from src.retrieval.embeddings import EmbeddingProvider, get_embedding_provider

#: Standard RRF constant (Cormack et al. 2009). No learned weights.
RRF_K = 60


def rrf(*result_lists: Sequence[RetrievedProduct], top_k: int = 20) -> list[RetrievedProduct]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    Args:
        *result_lists: Ranked candidate lists (best first).
        top_k: Number of fused results to return.

    Returns:
        Fused list, best first, each item tagged ``source="hybrid"`` with its RRF score.
    """
    scores: dict[int, float] = defaultdict(float)
    by_id: dict[int, RetrievedProduct] = {}
    for lst in result_lists:
        for rank, r in enumerate(lst, start=1):
            scores[r.product_id] += 1.0 / (RRF_K + rank)
            by_id.setdefault(r.product_id, r)
    ranked = sorted(scores.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[:top_k]
    return [
        RetrievedProduct(**{**by_id[pid].model_dump(), "score": float(s), "source": "hybrid"})
        for pid, s in ranked
    ]


class HybridRetriever:
    """BM25 + dense fused with RRF, truncated to top-k."""

    def __init__(
        self, bm25: BM25Index, dense: DenseStore, embedder: EmbeddingProvider | None = None
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self._embedder = embedder

    @property
    def embedder(self) -> EmbeddingProvider:
        """Embedding provider, resolved lazily from settings when not injected."""
        if self._embedder is None:
            self._embedder = get_embedding_provider()
        return self._embedder

    async def search(self, query: str, k: int = 20) -> list[RetrievedProduct]:
        """Run both retrievers and fuse them; empty query → empty result."""
        if not query.strip():
            return []
        bm25_hits = self.bm25.search(query, k=k)
        query_emb = await self.embedder.embed_query(query)
        dense_hits = await self.dense.search(query_emb, k=k)
        return rrf(bm25_hits, dense_hits, top_k=k)
