"""Reciprocal Rank Fusion correctness."""
from __future__ import annotations

from src.api.schemas import RetrievedProduct
from src.retrieval.hybrid import rrf


def _rp(pid: int, score: float = 1.0, source: str = "bm25") -> RetrievedProduct:
    return RetrievedProduct(
        product_id=pid, product_name=f"p{pid}", aisle="x", department="y",
        score=score, source=source,
    )


def test_rrf_merges_overlapping_lists() -> None:
    a = [_rp(1), _rp(2), _rp(3)]
    b = [_rp(2), _rp(1), _rp(4)]
    fused = rrf(a, b, top_k=5)
    # Product 1 and 2 appear in both, should be on top
    top_ids = [r.product_id for r in fused[:2]]
    assert set(top_ids) == {1, 2}


def test_rrf_handles_empty_inputs() -> None:
    fused = rrf([], [_rp(1)], top_k=5)
    assert len(fused) == 1
    assert fused[0].product_id == 1


def test_rrf_marks_as_hybrid() -> None:
    fused = rrf([_rp(1, source="bm25")], [_rp(1, source="dense")], top_k=5)
    assert fused[0].source == "hybrid"
