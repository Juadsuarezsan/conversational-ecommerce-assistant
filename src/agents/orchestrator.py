"""LangGraph orchestrator: intent_router → retriever → cart/refund handlers → synthesizer."""
from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.graph import END, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from src.agents.cart_manager import CartManager
from src.agents.intent_router import IntentRouter
from src.agents.refund_handler import evaluate_refund
from src.agents.synthesizer import Synthesizer
from src.api.schemas import CartItem, ChatResponse, IntentDecision, RetrievedProduct
from src.config import get_settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseStore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import get_reranker


class ConvState(TypedDict, total=False):
    session_id: str
    user_id: str
    message: str
    history: list[dict[str, str]]
    intent: IntentDecision
    retrieved: list[RetrievedProduct]
    reranked: list[RetrievedProduct]
    cart: list[CartItem]
    escalated: bool
    response: str
    trace_id: str
    started_at: float


def build_graph(*, bm25: BM25Index, dense: DenseStore, cart_mgr: CartManager) -> Any:
    settings = get_settings()
    router = IntentRouter(model=settings.anthropic_model, api_key=settings.anthropic_api_key)
    synth = Synthesizer(model=settings.anthropic_model, api_key=settings.anthropic_api_key)
    retriever = HybridRetriever(bm25=bm25, dense=dense)
    reranker = get_reranker()

    async def classify_node(state: ConvState) -> dict[str, Any]:
        decision = await router.classify(state["message"], history=state.get("history", []))
        logger.info(f"intent={decision.intent} confidence={decision.confidence:.2f}")
        return {"intent": decision}

    def route_after_classify(state: ConvState) -> str:
        intent = state["intent"].intent
        if intent == "escalate":
            return "escalate"
        if intent == "cart_op":
            return "cart"
        if intent == "refund":
            return "refund"
        if intent == "product_search":
            return "retrieve"
        return "synth"  # greeting / order_status (no retrieval)

    async def retrieve_node(state: ConvState) -> dict[str, Any]:
        hits = await retriever.search(state["message"], k=settings.max_retrieval_k)
        reranked = await reranker.rerank(state["message"], hits, top_n=settings.rerank_top_n)
        return {"retrieved": hits, "reranked": reranked}

    def cart_node(state: ConvState) -> dict[str, Any]:
        cart = cart_mgr.view(state["session_id"])
        return {"cart": cart}

    def refund_node(state: ConvState) -> dict[str, Any]:
        cart = cart_mgr.view(state["session_id"])
        total = sum(c.unit_price_usd * c.quantity for c in cart)
        decision = evaluate_refund(
            total_usd=total, department=None, threshold_usd=settings.escalation_refund_threshold_usd,
        )
        return {"cart": cart, "escalated": decision.requires_escalation}

    def escalate_node(state: ConvState) -> dict[str, Any]:
        return {"escalated": True}

    async def synth_node(state: ConvState) -> dict[str, Any]:
        text = await synth.respond(
            message=state["message"],
            intent=state["intent"].intent,
            products=state.get("reranked") or state.get("retrieved", []),
            cart=state.get("cart", []),
            escalated=state.get("escalated", False),
        )
        return {"response": text}

    g = StateGraph(ConvState)
    g.add_node("classify", classify_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("cart", cart_node)
    g.add_node("refund", refund_node)
    g.add_node("escalate", escalate_node)
    g.add_node("synth", synth_node)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_after_classify, {
        "retrieve": "retrieve", "cart": "cart", "refund": "refund",
        "escalate": "escalate", "synth": "synth",
    })
    g.add_edge("retrieve", "synth")
    g.add_edge("cart", "synth")
    g.add_edge("refund", "synth")
    g.add_edge("escalate", "synth")
    g.add_edge("synth", END)

    return g.compile()


async def run_turn(graph: Any, *, session_id: str, user_id: str, message: str,
                   history: list[dict[str, str]] | None = None) -> ChatResponse:
    started = time.perf_counter()
    state_in: ConvState = {
        "session_id": session_id, "user_id": user_id, "message": message,
        "history": history or [], "trace_id": str(uuid.uuid4()), "started_at": started,
    }
    state_out = await graph.ainvoke(state_in)
    latency_ms = int((time.perf_counter() - started) * 1000)
    decision: IntentDecision = state_out["intent"]
    return ChatResponse(
        session_id=session_id,
        turn=len(history) // 2 + 1 if history else 1,
        intent=decision.intent,
        confidence=decision.confidence,
        response=state_out.get("response", ""),
        retrieved_products=state_out.get("reranked", state_out.get("retrieved", [])),
        cart_snapshot=state_out.get("cart", []),
        escalated=state_out.get("escalated", False),
        latency_ms=latency_ms,
        trace_id=state_in["trace_id"],
    )
