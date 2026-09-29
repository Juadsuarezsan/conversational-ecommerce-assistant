"""IR metrics."""

from __future__ import annotations

import pytest

from src.eval.metrics import aggregate, mrr, ndcg_at_k, percentile, precision_at_k, recall_at_k


def test_precision_at_k() -> None:
    assert precision_at_k([1, 2, 3], {1: 3, 2: 2, 3: 1}, k=3) == 1.0
    assert precision_at_k([1, 99, 2, 98], {1: 3, 2: 2}, k=4) == 0.5
    assert precision_at_k([1], {1: 3}, k=0) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k([1, 2, 99], {1: 3, 2: 2}, k=10) == 1.0
    assert recall_at_k([1, 2, 3], {}, k=5) == 1.0
    assert recall_at_k([99], {1: 3, 2: 2}, k=5) == 0.0


def test_mrr() -> None:
    assert mrr([1, 99, 98], {1: 3}) == 1.0
    assert mrr([99, 98, 1], {1: 3}) == pytest.approx(1 / 3)
    assert mrr([99, 98, 97], {1: 3}) == 0.0


def test_ndcg() -> None:
    assert ndcg_at_k([1, 2], {1: 3, 2: 2}, k=2) == 1.0
    assert 0.0 < ndcg_at_k([2, 1], {1: 3, 2: 2}, k=2) < 1.0
    assert ndcg_at_k([9], {}, k=2) == 1.0
    assert ndcg_at_k([9], {1: 3}, k=2) == 0.0


def test_aggregate_and_percentile() -> None:
    assert aggregate([{"a": 1.0}, {"a": 3.0}]) == {"a": 2.0}
    assert aggregate([]) == {}
    assert percentile([], 50) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([5.0], 95) == 5.0
