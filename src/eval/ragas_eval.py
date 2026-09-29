"""RAGAS evaluation (Faithfulness, Answer Relevance, Context Precision, Context Recall).

Requires the ``ragas`` extra and ``ANTHROPIC_API_KEY`` (RAGAS uses the LLM as
judge and needs embeddings). Everything is imported lazily so the base install and
the tests never pull ``ragas``/``datasets``. :func:`build_ragas_rows` is pure and
tested; :func:`run_ragas` is the key-gated entry point.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.agents.orchestrator import build_graph, run_turn
from src.agents.runtime import AgentRuntime, build_runtime
from src.config import get_settings

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class RagasUnavailable(RuntimeError):
    """Raised when RAGAS cannot run (missing extra or key)."""


def build_ragas_rows(
    records: Sequence[dict[str, Any]],
    answers: Sequence[str],
    contexts: Sequence[Sequence[str]],
    catalog_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Assemble RAGAS rows: question, answer, contexts, ground_truth.

    ``ground_truth`` is the comma-separated names of the relevance-3 products so
    RAGAS can compute context recall against a textual reference.
    """
    rows: list[dict[str, Any]] = []
    for rec, ans, ctx in zip(records, answers, contexts, strict=True):
        gold = [
            catalog_names[int(r["product_id"])]
            for r in rec.get("relevant", [])
            if int(r["relevance"]) == 3 and int(r["product_id"]) in catalog_names
        ]
        rows.append(
            {
                "question": rec["query"],
                "answer": ans,
                "contexts": list(ctx),
                "ground_truth": ", ".join(gold) or "no exact match in catalog",
            }
        )
    return rows


async def collect_answers(
    records: Sequence[dict[str, Any]], rt: AgentRuntime | None = None
) -> tuple[list[str], list[list[str]]]:
    """Run each single-turn query through the agent and collect answer + retrieved contexts."""
    runtime = rt or await build_runtime()
    if runtime.graph is None:
        runtime.graph = build_graph(runtime)
    answers: list[str] = []
    contexts: list[list[str]] = []
    for rec in records:
        response, _ = await run_turn(
            runtime.graph, session_id=f"ragas-{rec['id']}", user_id="eval", message=rec["query"]
        )
        answers.append(response.response)
        contexts.append(
            [
                f"[pid:{p.product_id}] {p.product_name} ({p.aisle} / {p.department})"
                for p in response.retrieved_products
            ]
        )
    if rt is None:
        await runtime.close()
    return answers, contexts


async def run_ragas(
    records: Sequence[dict[str, Any]], rt: AgentRuntime | None = None
) -> dict[str, float]:
    """Compute the four RAGAS metrics. Raises :class:`RagasUnavailable` without key/extra."""
    s = get_settings()
    if not s.anthropic_api_key:
        raise RagasUnavailable("RAGAS requires ANTHROPIC_API_KEY")
    try:
        from datasets import Dataset
        from langchain_anthropic import ChatAnthropic
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RagasUnavailable("install the 'ragas' extra: pip install -e '.[ragas]'") from exc

    runtime = rt or await build_runtime()
    if runtime.graph is None:
        runtime.graph = build_graph(runtime)
    names = {int(p["product_id"]): str(p["product_name"]) for p in runtime.catalog}
    answers, contexts = await collect_answers(records, runtime)
    rows = build_ragas_rows(records, answers, contexts, names)
    llm = ChatAnthropic(model=s.anthropic_model, temperature=0.0, timeout=s.llm_timeout_seconds)
    result = ragas_evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
    )
    if rt is None:
        await runtime.close()
    return {m: float(result[m]) for m in RAGAS_METRICS}
