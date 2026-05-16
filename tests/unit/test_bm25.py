"""BM25 retrieval over the synthetic catalog."""
from __future__ import annotations


def test_bm25_finds_known_product(bm25_index) -> None:
    results = bm25_index.search("Heinz ketchup", k=5)
    assert len(results) > 0
    # First-page should contain at least one ketchup match (product_id 1 or 2)
    assert any(r.product_id in (1, 2) for r in results)


def test_bm25_empty_query_returns_empty(bm25_index) -> None:
    assert bm25_index.search("", k=5) == []
    assert bm25_index.search("   ", k=5) == []


def test_bm25_kid_cereal(bm25_index) -> None:
    results = bm25_index.search("kid cereal", k=5)
    ids = [r.product_id for r in results]
    # Cheerios Kids Snack Pack (3) or Cheerios Honey Nut Kid-Friendly (4)
    assert any(pid in (3, 4) for pid in ids)


def test_bm25_score_decreases(bm25_index) -> None:
    results = bm25_index.search("organic milk", k=5)
    if len(results) >= 2:
        assert results[0].score >= results[1].score
