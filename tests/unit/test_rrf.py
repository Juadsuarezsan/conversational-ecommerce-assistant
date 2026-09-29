"""Reciprocal Rank Fusion and the hybrid retriever."""

from __future__ import annotations

from typing import Any

from src.api.schemas import RetrievedProduct
from src.retrieval.dense import InMemoryDense
from src.retrieval.hybrid import HybridRetriever, rrf


def _rp(pid: int, score: float = 1.0, source: str = "bm25") -> RetrievedProduct:
    return RetrievedProduct(product_id=pid, product_name=f"p{pid}", aisle="x", department="y", score=score, source=source)  # type: ignore[arg-type]


def test_rrf_merges_overlapping_lists() -> None:
    fused = rrf([_rp(1), _rp(2), _rp(3)], [_rp(2), _rp(1), _rp(4)], top_k=5)
    assert {r.product_id for r in fused[:2]} == {1, 2}
    assert fused[0].source == "hybrid" and fused[0].score > fused[-1].score


def test_rrf_handles_empty_inputs_and_top_k() -> None:
    fused = rrf([], [_rp(1)], top_k=5)
    assert [r.product_id for r in fused] == [1]
    assert len(rrf([_rp(1), _rp(2), _rp(3)], top_k=2)) == 2


def test_rrf_deterministic_tie_break() -> None:
    fused = rrf([_rp(5)], [_rp(3)], top_k=5)
    assert [r.product_id for r in fused] == [3, 5]


async def test_hybrid_retriever(
    catalog: list[dict[str, Any]], bm25_index: Any, hash_embedder: Any
) -> None:
    dense = InMemoryDense(embedder=hash_embedder)
    await dense.index(catalog)
    retriever = HybridRetriever(bm25=bm25_index, dense=dense, embedder=hash_embedder)
    assert await retriever.search("   ") == []
    hits = await retriever.search("Heinz ketchup", k=10)
    assert hits and hits[0].product_id in (1, 2) and all(h.source == "hybrid" for h in hits)


async def test_hybrid_retriever_resolves_embedder_lazily(
    bm25_index: Any, catalog: list[dict[str, Any]]
) -> None:
    dense = InMemoryDense()
    await dense.index(catalog)
    retriever = HybridRetriever(bm25=bm25_index, dense=dense)
    assert retriever.embedder.name == "hash"
