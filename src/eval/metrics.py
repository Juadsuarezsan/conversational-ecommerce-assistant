"""Standard IR metrics: Precision@k, Recall@k, MRR, nDCG@k.

Each function takes the ranked list of retrieved ``product_id`` and the ground
truth as ``{product_id: relevance}`` with relevance in ``1..3``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    """Fraction of the top-``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for pid in top if pid in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    """Fraction of relevant items found in the top-``k`` (1.0 when nothing is relevant)."""
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    hits = sum(1 for pid in relevant if pid in top)
    return hits / len(relevant)


def mrr(retrieved: Sequence[int], relevant: dict[int, int]) -> float:
    """Reciprocal rank of the first relevant result (0.0 when none)."""
    for rank, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    """Discounted cumulative gain with ``2^rel - 1`` gains."""
    score = 0.0
    for i, pid in enumerate(retrieved[:k]):
        rel = relevant.get(pid, 0)
        if rel > 0:
            score += (2**rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    """DCG normalised by the ideal ordering (1.0 when nothing is relevant)."""
    if not relevant:
        return 1.0
    actual = dcg_at_k(retrieved, relevant, k)
    ideal_order = sorted(relevant.values(), reverse=True)
    ideal = 0.0
    for i, rel in enumerate(ideal_order[:k]):
        ideal += (2**rel - 1) / math.log2(i + 2)
    return actual / ideal if ideal > 0 else 0.0


def aggregate(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries (empty input → empty dict)."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(d[k] for d in per_query) / len(per_query) for k in keys}


def percentile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (``p`` in 0..100) of ``xs``; 0.0 when empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
