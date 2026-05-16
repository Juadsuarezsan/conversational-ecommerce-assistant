"""Hybrid search = BM25 + dense + Reciprocal Rank Fusion + (optional) reranker."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from src.api.schemas import RetrievedProduct
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseStore
from src.retrieval.embeddings import get_embedding_provider

RRF_K = 60


def rrf(*result_lists: Sequence[RetrievedProduct], top_k: int = 20) -> list[RetrievedProduct]:
    """Reciprocal Rank Fusion. Standard k=60; no learnable weights."""
    scores: dict[int, float] = defaultdict(float)
    by_id: dict[int, RetrievedProduct] = {}
    for lst in result_lists:
        for rank, r in enumerate(lst, start=1):
            scores[r.product_id] += 1.0 / (RRF_K + rank)
            if r.product_id not in by_id:
                by_id[r.product_id] = r
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        RetrievedProduct(**{**by_id[pid].model_dump(), "score": float(s), "source": "hybrid"})
        for pid, s in ranked
    ]


class HybridRetriever:
    def __init__(self, bm25: BM25Index, dense: DenseStore) -> None:
        self.bm25 = bm25
        self.dense = dense

    async def search(self, query: str, k: int = 20) -> list[RetrievedProduct]:
        if not query.strip():
            return []
        bm25_hits = self.bm25.search(query, k=k)
        embedder = get_embedding_provider()
        query_emb = await embedder.embed_query(query)
        dense_hits = await self.dense.search(query_emb, k=k)
        return rrf(bm25_hits, dense_hits, top_k=k)
