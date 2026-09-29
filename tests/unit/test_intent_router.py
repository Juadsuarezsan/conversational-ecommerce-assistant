"""Intent Router: heuristic fallback and the mocked Claude path."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.agents.intent_router import IntentRouter


def _router(api_key: str | None = None) -> IntentRouter:
    return IntentRouter(model="claude-sonnet-4-5-20250929", api_key=api_key)


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("I want a refund for my order", "refund"),
        ("I want to return the bread I bought yesterday", "refund"),
        ("Add 2 of those to my cart", "cart_op"),
        ("make that 3", "cart_op"),
        ("change ketchup quantity to 3", "cart_op"),
        ("Where is my order #1234", "order_status"),
        ("Get me a human agent please", "escalate"),
        ("this is unacceptable, get me your manager", "escalate"),
        ("hi", "greeting"),
        ("thanks!", "greeting"),
        ("show me healthy breakfast options", "product_search"),
    ],
)
async def test_heuristic_intents(message: str, intent: str) -> None:
    decision, usage = await _router().classify(message)
    assert decision.intent == intent
    assert usage.llm_calls == 0 and usage.cost_usd == 0.0


async def test_llm_path_parses_json_and_tracks_usage(mocker: Any, fake_chat_factory: Any) -> None:
    chat = fake_chat_factory(
        [
            '```json\n{"intent": "product_search", "confidence": 0.93, "extracted_filters": {"price_max": 5}, "rationale": "x"}\n```'
        ]
    )
    router = _router(api_key="sk-test")
    mocker.patch.object(IntentRouter, "_build_chat", return_value=chat)
    decision, usage = await router.classify(
        "cereal under $5", history=[{"role": "user", "content": "hi"}]
    )
    assert decision.intent == "product_search" and decision.extracted_filters["price_max"] == 5
    assert usage.llm_calls == 1 and usage.input_tokens == 120 and usage.cost_usd > 0
    assert "<conversation>" in chat.calls[0][1].content


async def test_llm_path_retries_then_raises_on_invalid_json(
    mocker: Any, fake_chat_factory: Any
) -> None:
    chat = fake_chat_factory(["not json", "still not json", "nope"])
    router = _router(api_key="sk-test")
    mocker.patch.object(IntentRouter, "_build_chat", return_value=chat)
    mocker.patch.object(IntentRouter._classify_llm.retry, "sleep", AsyncMock())
    with pytest.raises(ValueError):
        await router.classify("anything")
    assert len(chat.calls) == 3


async def test_llm_path_recovers_after_transient_error(mocker: Any, fake_chat_factory: Any) -> None:
    chat = fake_chat_factory(["garbage", '{"intent": "greeting", "confidence": 0.9}'])
    router = _router(api_key="sk-test")
    mocker.patch.object(IntentRouter, "_build_chat", return_value=chat)
    mocker.patch.object(IntentRouter._classify_llm.retry, "sleep", AsyncMock())
    decision, _ = await router.classify("hello")
    assert decision.intent == "greeting"


def test_strip_fences() -> None:
    assert IntentRouter._strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert IntentRouter._strip_fences('{"a": 1}') == '{"a": 1}'


def test_build_chat_constructs_client() -> None:
    chat = _router(api_key="sk-test")._build_chat()
    assert chat.model == "claude-sonnet-4-5-20250929"
    assert chat.max_retries == 0
