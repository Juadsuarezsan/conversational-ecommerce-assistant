"""Unit tests for IR metrics."""
from __future__ import annotations

from src.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_at_k_perfect() -> None:
    assert precision_at_k([1, 2, 3], {1: 3, 2: 2, 3: 1}, k=3) == 1.0


def test_precision_at_k_half() -> None:
    assert precision_at_k([1, 99, 2, 98], {1: 3, 2: 2}, k=4) == 0.5


def test_recall_at_k_full() -> None:
    assert recall_at_k([1, 2, 99], {1: 3, 2: 2}, k=10) == 1.0


def test_recall_at_k_no_relevant_is_one() -> None:
    # Trivially perfect when there's nothing to recall
    assert recall_at_k([1, 2, 3], {}, k=5) == 1.0


def test_mrr_first_position() -> None:
    assert mrr([1, 99, 98], {1: 3}) == 1.0


def test_mrr_third_position() -> None:
    assert mrr([99, 98, 1], {1: 3}) == 1.0 / 3


def test_mrr_no_match() -> None:
    assert mrr([99, 98, 97], {1: 3}) == 0.0


def test_ndcg_at_k_perfect() -> None:
    # Retrieved exactly the most-relevant first
    assert ndcg_at_k([1, 2], {1: 3, 2: 2}, k=2) == 1.0


def test_ndcg_at_k_inverted() -> None:
    score = ndcg_at_k([2, 1], {1: 3, 2: 2}, k=2)
    assert 0.0 < score < 1.0
