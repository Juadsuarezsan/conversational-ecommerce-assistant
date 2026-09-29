"""LangGraph orchestrator: classify → (retrieve | cart | refund | order | escalate) → synth.

Every node is wrapped by :func:`src.observability.node_logger`, so the input and
output state of each step is logged with the request ``trace_id``. Token usage and
cost accumulate in ``state["usage"]``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from src.agents.cart_ops import execute_operation, parse_operation, resolve_product
from src.agents.intent_router import IntentRouter
from src.agents.orders import extract_order_id
from src.agents.refund_handler import evaluate_refund, extract_amount
from src.agents.runtime import AgentRuntime
from src.agents.synthesizer import Synthesizer
from src.api.schemas import (
    CartAction,
    CartItem,
    ChatResponse,
    IntentDecision,
    RetrievedProduct,
    Usage,
)
from src.observability import node_logger
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.structured import (
    StructuredFilter,
    apply_filter,
    filters_from_router,
    lexically_related,
    parse_filters,
)
from src.session.cache import SessionState


class ConvState(TypedDict, total=False):
    """Mutable state threaded through the graph for one turn."""

    session_id: str
    user_id: str
    message: str
    history: list[dict[str, str]]
    last_products: list[RetrievedProduct]
    intent: IntentDecision
    filters: StructuredFilter
    retrieved: list[RetrievedProduct]
    reranked: list[RetrievedProduct]
    cart: list[CartItem]
    cart_action: CartAction | None
    note: str
    escalated: bool
    response: str
    usage: Usage
    trace_id: str
    started_at: float


def route_after_classify(state: ConvState) -> str:
    """Pick the branch after intent classification."""
    intent = state["intent"].intent
    return {
        "escalate": "escalate_to_human",
        "cart_op": "cart_ops",
        "refund": "refund_policy",
        "product_search": "retrieve_products",
        "order_status": "order_lookup",
    }.get(intent, "synthesize")


def build_graph(rt: AgentRuntime) -> Any:
    """Compile the conversation graph on top of an :class:`AgentRuntime`."""
    s = rt.settings
    router = IntentRouter(
        model=s.anthropic_model, api_key=s.anthropic_api_key, timeout=s.llm_timeout_seconds
    )
    synth = Synthesizer(
        model=s.anthropic_model, api_key=s.anthropic_api_key, timeout=s.llm_timeout_seconds
    )
    retriever = HybridRetriever(bm25=rt.bm25, dense=rt.dense, embedder=rt.embedder)

    @node_logger("classify")
    async def classify_node(state: ConvState) -> dict[str, Any]:
        decision, usage = await router.classify(state["message"], history=state.get("history", []))
        filters = parse_filters(state["message"]).merged(
            filters_from_router(decision.extracted_filters)
        )
        return {
            "intent": decision,
            "filters": filters,
            "usage": state.get("usage", Usage()).add(usage),
        }

    @node_logger("retrieve")
    async def retrieve_node(state: ConvState) -> dict[str, Any]:
        query = state["message"]
        hits = await retriever.search(query, k=s.max_retrieval_k)
        pool = await rt.reranker.rerank(query, hits, top_n=max(2 * s.rerank_top_n, 10))
        reranked = pool[: s.rerank_top_n]
        f = state.get("filters", StructuredFilter())
        note = ""
        if not f.is_empty():
            # Sort/filter the reranked pool first (keeps relevance), then widen to all
            # fused hits, then fall back to a pure catalog filter.
            filtered = apply_filter(lexically_related(query, pool), f) or apply_filter(
                lexically_related(query, hits), f
            )
            if filtered:
                reranked = filtered[: s.rerank_top_n]
            else:
                reranked = await rt.structured.search(f, k=s.rerank_top_n)
                note = "No semantic match satisfied the filters; showing catalog items that do."
        return {"retrieved": hits, "reranked": reranked, "note": note}

    @node_logger("cart")
    async def cart_node(state: ConvState) -> dict[str, Any]:
        sid = state["session_id"]
        cart = await rt.cart_mgr.view(sid)
        op = parse_operation(state["message"], state.get("last_products", []), cart, rt.bm25)
        result = await execute_operation(
            op,
            session_id=sid,
            cart_mgr=rt.cart_mgr,
            last_products=state.get("last_products", []),
            cart=cart,
            bm25=rt.bm25,
        )
        return {"cart": result.cart, "cart_action": op.action, "note": result.message}

    @node_logger("refund")
    async def refund_node(state: ConvState) -> dict[str, Any]:
        sid = state["session_id"]
        cart = await rt.cart_mgr.view(sid)
        departments: list[str] = []
        amount = extract_amount(state["message"])
        order_id = extract_order_id(state["message"])
        order = await rt.orders.get(order_id) if order_id else None
        if order is not None:
            amount = amount if amount is not None else order.total_usd
            if order.department:
                departments.append(order.department)
        named = resolve_product(
            state["message"], state.get("last_products", []), cart, rt.bm25, prefer_cart=True
        )
        if named is not None:
            row = rt.bm25.get(named.product_id)
            if row is not None:
                departments.append(str(row.get("department", "")))
            if amount is None:
                price = (
                    named.unit_price_usd
                    if isinstance(named, CartItem)
                    else float(named.price_usd or 0.0)
                )
                amount = price * (named.quantity if isinstance(named, CartItem) else 1)
        if amount is None:
            amount = sum(c.unit_price_usd * c.quantity for c in cart)
            for c in cart:
                row = rt.bm25.get(c.product_id)
                if row is not None:
                    departments.append(str(row.get("department", "")))
        decision = evaluate_refund(
            total_usd=amount,
            departments=departments,
            threshold_usd=s.escalation_refund_threshold_usd,
        )
        if decision.requires_escalation:
            note = f"Refund of ${amount:.2f} needs a human: {decision.reason}."
        else:
            note = f"Refund of ${amount:.2f} approved automatically ({decision.reason})."
        return {"cart": cart, "escalated": decision.requires_escalation, "note": note}

    @node_logger("order")
    async def order_node(state: ConvState) -> dict[str, Any]:
        order_id = extract_order_id(state["message"])
        order = (
            await rt.orders.get(order_id)
            if order_id
            else await rt.orders.latest_for_user(state["user_id"])
        )
        if order is None:
            return {"note": "I couldn't find that order. Could you share the order number?"}
        items = ", ".join(order.items)
        return {
            "note": f"Order #{order.order_id} ({items}) is {order.status}. Total ${order.total_usd:.2f}."
        }

    @node_logger("escalate")
    async def escalate_node(state: ConvState) -> dict[str, Any]:
        return {"escalated": True, "note": "customer asked for a human"}

    @node_logger("synth")
    async def synth_node(state: ConvState) -> dict[str, Any]:
        text, usage = await synth.respond(
            message=state["message"],
            intent=state["intent"].intent,
            products=state.get("reranked") or state.get("retrieved", []),
            cart=state.get("cart", []),
            escalated=state.get("escalated", False),
            note=state.get("note", ""),
        )
        return {"response": text, "usage": state.get("usage", Usage()).add(usage)}

    # Node names must not collide with state keys (LangGraph raises otherwise).
    g = StateGraph(ConvState)
    g.add_node("classify_intent", classify_node)
    g.add_node("retrieve_products", retrieve_node)
    g.add_node("cart_ops", cart_node)
    g.add_node("refund_policy", refund_node)
    g.add_node("order_lookup", order_node)
    g.add_node("escalate_to_human", escalate_node)
    g.add_node("synthesize", synth_node)
    g.set_entry_point("classify_intent")
    branches = [
        "retrieve_products",
        "cart_ops",
        "refund_policy",
        "order_lookup",
        "escalate_to_human",
    ]
    g.add_conditional_edges(
        "classify_intent", route_after_classify, {b: b for b in [*branches, "synthesize"]}
    )
    for node in branches:
        g.add_edge(node, "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


async def run_turn(
    graph: Any,
    *,
    session_id: str,
    user_id: str,
    message: str,
    session: SessionState | None = None,
    llm_enabled: bool = False,
) -> tuple[ChatResponse, ConvState]:
    """Run one conversational turn and return the response plus the final graph state."""
    started = time.perf_counter()
    sess = session or SessionState()
    trace_id = str(uuid.uuid4())
    state_in: ConvState = {
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
        "history": sess.history,
        "last_products": sess.last_products,
        "trace_id": trace_id,
        "started_at": started,
        "usage": Usage(),
    }
    state_out: ConvState = await graph.ainvoke(
        state_in,
        config={
            "run_name": "chat_turn",
            "tags": ["ecommerce-assistant"],
            "metadata": {"trace_id": trace_id},
        },
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    decision = state_out["intent"]
    usage = state_out.get("usage", Usage())
    logger.bind(trace_id=trace_id).info(
        "turn done intent={} latency_ms={} tokens_in={} tokens_out={} cost_usd={:.6f}",
        decision.intent,
        latency_ms,
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )
    response = ChatResponse(
        session_id=session_id,
        turn=sess.turn + 1,
        intent=decision.intent,
        confidence=decision.confidence,
        response=state_out.get("response", ""),
        retrieved_products=state_out.get("reranked", state_out.get("retrieved", [])),
        cart_snapshot=state_out.get("cart", []),
        cart_action=state_out.get("cart_action"),
        escalated=state_out.get("escalated", False),
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
        llm_used=llm_enabled and usage.llm_calls > 0,
        trace_id=trace_id,
    )
    return response, state_out


def next_session_state(
    prev: SessionState, response: ChatResponse, state_out: ConvState
) -> SessionState:
    """Fold a finished turn into the session: history grows, last products update on searches."""
    from src.session.cache import append_turn

    new = append_turn(prev, state_out["message"], response.response)
    shown: Sequence[RetrievedProduct] = state_out.get("reranked") or []
    if response.intent == "product_search" and shown:
        new = new.model_copy(update={"last_products": list(shown)})
    return new
