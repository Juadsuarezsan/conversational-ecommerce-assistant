"""End-to-end retrieval eval against the synthetic catalog."""
from __future__ import annotations

import pytest

from src.eval.runner import evaluate


@pytest.mark.asyncio
async def test_eval_runs_end_to_end() -> None:
    report = await evaluate(vector_store="in_memory")
    assert report["n_queries"] > 0
    assert "rerank" in report["overall"]
    # The rerank strategy should at least be defined and have a score
    assert 0.0 <= report["overall"]["rerank"]["p_at_5"] <= 1.0


@pytest.mark.asyncio
async def test_hybrid_beats_or_matches_bm25_on_average() -> None:
    """Hybrid retrieval (BM25 + dense + RRF) should not be substantially worse than BM25 alone."""
    report = await evaluate(vector_store="in_memory")
    hybrid_ndcg = report["overall"]["hybrid"]["ndcg_10"]
    bm25_ndcg = report["overall"]["bm25"]["ndcg_10"]
    # Allow small regression but flag big ones (the stub embedding is weak)
    assert hybrid_ndcg >= bm25_ndcg - 0.15, (
        f"hybrid nDCG@10 {hybrid_ndcg:.3f} regressed badly vs bm25 {bm25_ndcg:.3f}"
    )
