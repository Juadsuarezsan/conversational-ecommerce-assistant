"""LangGraph orchestrator: every branch through the offline runtime."""

from __future__ import annotations

from typing import Any

from src.agents.orchestrator import next_session_state, route_after_classify, run_turn
from src.api.schemas import IntentDecision
from src.session.cache import SessionState


def test_route_after_classify() -> None:
    def st(intent: str) -> Any:
        return {"intent": IntentDecision(intent=intent, confidence=1.0)}  # type: ignore[arg-type]

    assert route_after_classify(st("product_search")) == "retrieve_products"
    assert route_after_classify(st("cart_op")) == "cart_ops"
    assert route_after_classify(st("refund")) == "refund_policy"
    assert route_after_classify(st("order_status")) == "order_lookup"
    assert route_after_classify(st("escalate")) == "escalate_to_human"
    assert route_after_classify(st("greeting")) == "synthesize"


async def test_search_then_cart_flow(runtime: Any) -> None:
    session = SessionState()
    r1, s1 = await run_turn(
        runtime.graph,
        session_id="s",
        user_id="u",
        message="lactose free milk gallon",
        session=session,
    )
    assert r1.intent == "product_search" and r1.retrieved_products and r1.trace_id and r1.turn == 1
    assert r1.llm_used is False and r1.cost_usd == 0.0
    session = next_session_state(session, r1, s1)
    assert [p.product_id for p in session.last_products] == [
        p.product_id for p in r1.retrieved_products
    ]
    second = r1.retrieved_products[1].product_id
    r2, s2 = await run_turn(
        runtime.graph,
        session_id="s",
        user_id="u",
        message="add 2 of the second one to my cart",
        session=session,
    )
    assert r2.intent == "cart_op" and r2.cart_action == "add" and r2.turn == 2
    assert [(c.product_id, c.quantity) for c in r2.cart_snapshot] == [(second, 2)]
    session = next_session_state(session, r2, s2)
    assert len(session.history) == 4 and session.last_products  # unchanged by cart ops


async def test_filters_sort_and_fallback_to_catalog(runtime: Any) -> None:
    r, _ = await run_turn(runtime.graph, session_id="s", user_id="u", message="cheapest coffee")
    assert r.retrieved_products[0].product_id == 124
    r, _ = await run_turn(
        runtime.graph, session_id="s", user_id="u", message="top rated cereals under $5"
    )
    assert r.retrieved_products[0].product_id == 4 and all(
        p.price_usd <= 5 for p in r.retrieved_products
    )
    r, _ = await run_turn(runtime.graph, session_id="s", user_id="u", message="zzqx under $2")
    assert r.retrieved_products and all(p.price_usd <= 2 for p in r.retrieved_products)
    assert "catalog items" in r.response


async def test_refund_branches(runtime: Any) -> None:
    r, _ = await run_turn(
        runtime.graph, session_id="r1", user_id="u", message="refund my $200 order from last week"
    )
    assert r.intent == "refund" and r.escalated and "human" in r.response
    r, _ = await run_turn(runtime.graph, session_id="r2", user_id="u", message="refund order #1003")
    assert r.escalated and "alcohol" in r.response
    r, _ = await run_turn(
        runtime.graph, session_id="r3", user_id="u", message="I want to return the ketchup I bought"
    )
    assert not r.escalated and "approved" in r.response
    session = SessionState()
    r0, s0 = await run_turn(
        runtime.graph,
        session_id="r4",
        user_id="u",
        message="add [pid:210] to my cart",
        session=session,
    )
    session = next_session_state(session, r0, s0)
    r, _ = await run_turn(
        runtime.graph, session_id="r4", user_id="u", message="return my order", session=session
    )
    assert r.escalated and "alcohol" in r.response  # cart-based amount + department


async def test_order_escalate_greeting(runtime: Any) -> None:
    r, _ = await run_turn(
        runtime.graph, session_id="o", user_id="demo-user", message="where is my order #1001"
    )
    assert r.intent == "order_status" and "1001" in r.response and "shipped" in r.response
    r, _ = await run_turn(
        runtime.graph, session_id="o", user_id="demo-user", message="where is my order"
    )
    assert "1003" in r.response  # latest order for the user
    r, _ = await run_turn(
        runtime.graph, session_id="o", user_id="ghost", message="where is my order"
    )
    assert "couldn't find" in r.response
    r, _ = await run_turn(
        runtime.graph, session_id="o", user_id="u", message="I want to speak to a human agent"
    )
    assert r.intent == "escalate" and r.escalated
    r, _ = await run_turn(runtime.graph, session_id="o", user_id="u", message="hi")
    assert r.intent == "greeting" and r.response.startswith("Hi!")
