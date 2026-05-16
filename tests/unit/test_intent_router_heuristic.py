"""Intent router heuristic fallback (no API key required)."""
from __future__ import annotations

from src.agents.intent_router import IntentRouter


def _router() -> IntentRouter:
    return IntentRouter(model="claude-sonnet-4-5", api_key=None)


async def test_refund_keyword() -> None:
    d = await _router().classify("I want a refund for my order")
    assert d.intent == "refund"


async def test_cart_keyword() -> None:
    d = await _router().classify("Add 2 of those to my cart")
    assert d.intent == "cart_op"


async def test_order_status_keyword() -> None:
    d = await _router().classify("Where is my order #1234")
    assert d.intent == "order_status"


async def test_explicit_escalation() -> None:
    d = await _router().classify("Get me a human agent please")
    assert d.intent == "escalate"
    assert d.confidence >= 0.8


async def test_greeting() -> None:
    d = await _router().classify("hi")
    assert d.intent == "greeting"


async def test_default_is_product_search() -> None:
    d = await _router().classify("show me healthy breakfast options")
    assert d.intent == "product_search"
