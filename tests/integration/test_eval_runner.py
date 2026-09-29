"""Retrieval ablation, conversation eval and the `python -m eval.run` entry point (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.conversation import evaluate_conversations
from src.eval.runner import evaluate, format_report, load_eval_set
from src.eval.runner import main as runner_main


async def test_eval_runs_end_to_end_with_hash_embeddings(hash_embedder: Any) -> None:
    report = await evaluate(vector_store="in_memory", embedder=hash_embedder)
    assert report["n_queries"] == 80 and set(report["overall"]) == {
        "bm25",
        "dense",
        "hybrid",
        "rerank",
    }
    assert report["provenance"]["deterministic_fallback"] is True
    assert "sin modelo" in report["provenance"]["label"]
    assert 0.0 <= report["overall"]["rerank"]["p_at_5"] <= 1.0
    assert report["overall"]["bm25"]["ndcg_10"] > 0.5  # lookup queries are easy for BM25
    assert set(report["per_category"]) == {"lookup", "semantic", "comparative"}
    assert len(report["per_query"]) == 80 and "top" in report["per_query"][0]["rerank"]
    text = format_report(report)
    assert "E-COMMERCE RETRIEVAL EVAL" in text and "[lookup]" in text


async def test_hybrid_does_not_regress_badly_vs_bm25(hash_embedder: Any) -> None:
    subset = [q for q in load_eval_set() if q["category"] == "lookup"][:10]
    report = await evaluate(vector_store="in_memory", eval_set=subset, embedder=hash_embedder)
    assert report["overall"]["hybrid"]["ndcg_10"] >= report["overall"]["bm25"]["ndcg_10"] - 0.15


async def test_conversation_eval(runtime: Any) -> None:
    records = [r for r in load_eval_set() if r["category"] == "multi_turn"][:5]
    out = await evaluate_conversations(records, runtime)
    assert out["n_conversations"] == 5 and out["avg_turns"] >= 2 and out["llm_enabled"] is False
    assert "sin LLM" in out["label"] and out["avg_cost_usd"] == 0.0
    assert out["conversations"][0]["turns"][-1]["cart"] is not None


def test_runner_cli(capsys: Any) -> None:
    assert runner_main(["--gate", "0.0"]) == 0
    assert "Strategy" in capsys.readouterr().out
    assert runner_main(["--json", "--gate", "1.1"]) == 1
    assert json.loads(capsys.readouterr().out)["n_queries"] == 80


def test_eval_run_entry_point_writes_artifacts(tmp_path: Path) -> None:
    from eval.run import main as eval_main

    results = tmp_path / "RESULTS.md"
    assert (
        eval_main(
            [
                "--stores",
                "in_memory",
                "--out-dir",
                str(tmp_path / "runs"),
                "--results",
                str(results),
                "--name",
                "t",
            ]
        )
        == 0
    )
    runs = list((tmp_path / "runs").glob("*-t.json"))
    assert len(runs) == 1
    doc = json.loads(runs[0].read_text())
    assert (
        doc["retrieval"]["in_memory"]["n_queries"] == 80
        and doc["conversations"]["n_conversations"] == 20
    )
    assert "pending" in doc["ragas"] and "pending" in doc["zero_shot"] and "pending" in doc["judge"]
    md = results.read_text()
    assert "fallback determinista" in md and "pendiente (requiere ANTHROPIC_API_KEY)" in md
    assert "| qdrant | pendiente" in md and "Los 10 peores casos" in md
