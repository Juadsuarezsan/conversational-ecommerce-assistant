"""Multi-turn evaluation: scripted conversations through the full agent.

Each ``multi_turn`` record in ``data/eval/queries.jsonl`` carries ``setup_turns``
(sent first, same session) and an ``expect`` block that is checked after the
final query. The check is deterministic (intent, escalation, cart state, cited
products), which gives a task-completion rate that does not need an LLM judge.
The LLM-as-judge rubric (:mod:`src.eval.judge`) is the complementary,
key-gated measure.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from src.agents.orchestrator import build_graph, next_session_state, run_turn
from src.agents.runtime import AgentRuntime, build_runtime
from src.api.schemas import ChatResponse
from src.eval.metrics import percentile
from src.session.cache import SessionState

_PID_RE = re.compile(r"\[pid:(\d+)\]")


def check_expectations(
    expect: dict[str, Any],
    response: ChatResponse,
    previous_products: Sequence[int],
) -> list[str]:
    """Return the list of failed expectation names (empty when the turn passes)."""
    failed: list[str] = []
    cart_ids = [c.product_id for c in response.cart_snapshot]
    cart_qty = {c.product_id: c.quantity for c in response.cart_snapshot}
    cited = {int(m) for m in _PID_RE.findall(response.response)}

    if "intent" in expect and response.intent != expect["intent"]:
        failed.append("intent")
    if "escalated" in expect and response.escalated != bool(expect["escalated"]):
        failed.append("escalated")
    if "cart_has" in expect and not all(pid in cart_ids for pid in expect["cart_has"]):
        failed.append("cart_has")
    if "cart_has_any" in expect and not any(pid in cart_ids for pid in expect["cart_has_any"]):
        failed.append("cart_has_any")
    if "cart_has_rank" in expect:
        rank = int(expect["cart_has_rank"])
        target = previous_products[rank - 1] if 0 < rank <= len(previous_products) else None
        if target is None or target not in cart_ids:
            failed.append("cart_has_rank")
        elif "qty_rank" in expect and cart_qty.get(target) != int(expect["qty_rank"]):
            failed.append("qty_rank")
    if "cart_qty_rank" in expect:
        spec = expect["cart_qty_rank"]
        rank = int(spec["rank"])
        target = previous_products[rank - 1] if 0 < rank <= len(previous_products) else None
        if target is None or cart_qty.get(target) != int(spec["qty"]):
            failed.append("cart_qty_rank")
    if "cart_size" in expect and len(cart_ids) != int(expect["cart_size"]):
        failed.append("cart_size")
    if expect.get("cart_empty") and cart_ids:
        failed.append("cart_empty")
    if "response_contains" in expect and not all(
        s.lower() in response.response.lower() for s in expect["response_contains"]
    ):
        failed.append("response_contains")
    if "response_mentions_any" in expect and not (cited & set(expect["response_mentions_any"])):
        failed.append("response_mentions_any")
    return failed


async def run_conversation(
    rt: AgentRuntime, record: dict[str, Any], session_id: str
) -> dict[str, Any]:
    """Play one scripted conversation and return its outcome."""
    session = SessionState()
    turns: list[dict[str, Any]] = []
    total_cost = 0.0
    shown_before_final: list[int] = []
    t0 = time.perf_counter()
    messages = [*record.get("setup_turns", []), record["query"]]
    response: ChatResponse | None = None
    for message in messages:
        shown_before_final = [p.product_id for p in session.last_products]
        response, state_out = await run_turn(
            rt.graph,
            session_id=session_id,
            user_id="eval-user",
            message=message,
            session=session,
            llm_enabled=rt.settings.llm_enabled,
        )
        session = next_session_state(session, response, state_out)
        total_cost += response.cost_usd
        turns.append(
            {
                "user": message,
                "assistant": response.response,
                "intent": response.intent,
                "cart": [(c.product_id, c.quantity) for c in response.cart_snapshot],
                "escalated": response.escalated,
                "latency_ms": response.latency_ms,
            }
        )
    assert response is not None
    failed = check_expectations(record.get("expect", {}), response, shown_before_final)
    await rt.cart_mgr.clear(session_id)
    return {
        "id": record["id"],
        "query": record["query"],
        "turns": turns,
        "n_turns": len(turns),
        "passed": not failed,
        "failed_checks": failed,
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "cost_usd": round(total_cost, 6),
    }


async def evaluate_conversations(
    records: Sequence[dict[str, Any]], rt: AgentRuntime | None = None
) -> dict[str, Any]:
    """Run every ``multi_turn`` record and aggregate task completion, turns, latency, cost."""
    runtime = rt or await build_runtime()
    if runtime.graph is None:
        runtime.graph = build_graph(runtime)
    outcomes = [
        await run_conversation(runtime, rec, session_id=f"eval-{rec['id']}")
        for rec in records
        if rec.get("category") == "multi_turn"
    ]
    if rt is None:
        await runtime.close()
    passed = sum(1 for o in outcomes if o["passed"])
    latencies = [o["latency_ms"] for o in outcomes]
    return {
        "n_conversations": len(outcomes),
        "task_completion_rate": passed / len(outcomes) if outcomes else 0.0,
        "avg_turns": sum(o["n_turns"] for o in outcomes) / len(outcomes) if outcomes else 0.0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "avg_cost_usd": sum(o["cost_usd"] for o in outcomes) / len(outcomes) if outcomes else 0.0,
        "llm_enabled": runtime.settings.llm_enabled,
        "label": (
            "modelo de produccion"
            if runtime.settings.llm_enabled
            else "fallback determinista, sin LLM (router heuristico + plantillas)"
        ),
        "conversations": outcomes,
    }
