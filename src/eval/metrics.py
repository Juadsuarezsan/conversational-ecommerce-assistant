"""Standard IR metrics: Precision@k, Recall@k, MRR, nDCG@k.

Each function takes a ranked list of `product_id` and a dict of
`{product_id: relevance_score}` for the ground truth.
"""
from __future__ import annotations

import math
from typing import Sequence


def precision_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for pid in top if pid in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    if not relevant:
        return 1.0  # nothing to recall is trivially perfect
    top = set(retrieved[:k])
    hits = sum(1 for pid in relevant if pid in top)
    return hits / len(relevant)


def mrr(retrieved: Sequence[int], relevant: dict[int, int]) -> float:
    for rank, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    score = 0.0
    for i, pid in enumerate(retrieved[:k]):
        rel = relevant.get(pid, 0)
        if rel > 0:
            score += (2**rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    if not relevant:
        return 1.0
    actual = dcg_at_k(retrieved, relevant, k)
    ideal_order = sorted(relevant.values(), reverse=True)
    ideal = 0.0
    for i, rel in enumerate(ideal_order[:k]):
        ideal += (2**rel - 1) / math.log2(i + 2)
    return actual / ideal if ideal > 0 else 0.0


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(d[k] for d in per_query) / len(per_query) for k in keys}
