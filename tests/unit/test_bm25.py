"""BM25 retrieval over the synthetic catalog."""

from __future__ import annotations

from src.retrieval.bm25 import BM25Index, product_text, tokenize


def test_bm25_finds_known_product(bm25_index: BM25Index) -> None:
    results = bm25_index.search("Heinz ketchup", k=5)
    assert results
    assert any(r.product_id in (1, 2) for r in results)
    assert results[0].source == "bm25"


def test_bm25_empty_query_returns_empty(bm25_index: BM25Index) -> None:
    assert bm25_index.search("", k=5) == []
    assert bm25_index.search("   ", k=5) == []


def test_bm25_kid_cereal(bm25_index: BM25Index) -> None:
    ids = [r.product_id for r in bm25_index.search("kid cereal", k=5)]
    assert any(pid in (3, 4) for pid in ids)


def test_bm25_scores_are_descending(bm25_index: BM25Index) -> None:
    results = bm25_index.search("organic milk", k=5)
    assert len(results) >= 2
    assert results[0].score >= results[1].score


def test_bm25_plural_folding(bm25_index: BM25Index) -> None:
    """'cereals' must match products indexed as 'cereal'."""
    ids = [r.product_id for r in bm25_index.search("cereals", k=5)]
    assert 52 in ids or 4 in ids


def test_tokenize_and_stem() -> None:
    assert tokenize("Cheerios 18oz Cereals, glass!") == ["cheerio", "18", "oz", "cereal", "glass"]


def test_get_and_len(bm25_index: BM25Index) -> None:
    assert len(bm25_index) == 213
    assert bm25_index.get(1)["product_name"].startswith("Heinz")
    assert bm25_index.get(9999) is None


def test_product_text_includes_aisle_and_department() -> None:
    assert product_text({"product_name": "A", "aisle": "b", "department": "c"}) == "A b c"
