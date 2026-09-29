"""Orders store, observability helpers and settings."""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from src.agents.orders import InMemoryOrderStore, extract_order_id
from src.api.schemas import Usage
from src.config import DEFAULT_ANTHROPIC_MODEL, get_settings
from src.observability import (
    configure_logging,
    configure_tracing,
    node_logger,
    price_usd,
    usage_from_response,
)


async def test_order_store() -> None:
    store = InMemoryOrderStore()
    assert (await store.get("#1001")).status == "shipped"
    assert await store.get("9") is None
    assert (await store.latest_for_user("demo-user")).order_id == "1003"
    assert await store.latest_for_user("nobody") is None
    assert extract_order_id("where is my order #1001?") == "1001"
    assert extract_order_id("add 2 items") is None


def test_price_and_usage() -> None:
    assert price_usd(1_000_000, 0) == 3.0
    assert price_usd(0, 1_000_000) == 15.0
    u = usage_from_response(
        AIMessage(
            content="x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )
    )
    assert u.llm_calls == 1 and u.input_tokens == 10 and u.cost_usd == pytest.approx(0.000105)
    u2 = usage_from_response(
        AIMessage(content="x", response_metadata={"usage": {"input_tokens": 3, "output_tokens": 1}})
    )
    assert u2.input_tokens == 3
    assert usage_from_response(object()).input_tokens == 0
    total = Usage(input_tokens=1, cost_usd=0.1, llm_calls=1).add(
        Usage(output_tokens=2, cost_usd=0.2, llm_calls=1)
    )
    assert (total.input_tokens, total.output_tokens, total.llm_calls) == (
        1,
        2,
        2,
    ) and total.cost_usd == pytest.approx(0.3)


async def test_node_logger_wraps_sync_and_async() -> None:
    @node_logger("a")
    def sync_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"response": "ok", "retrieved": [], "cart": []}

    @node_logger("b")
    async def async_node(state: dict[str, Any]) -> None:
        return None

    assert await sync_node({"trace_id": "t", "message": "m" * 200}) == {
        "response": "ok",
        "retrieved": [],
        "cart": [],
    }
    assert await async_node({}) == {}


def test_configure_tracing_only_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert configure_tracing() is False
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    get_settings.cache_clear()
    assert configure_tracing() is True
    assert (
        os.environ["LANGCHAIN_TRACING_V2"] == "true"
        and os.environ["LANGCHAIN_PROJECT"] == "ecommerce-assistant"
    )
    monkeypatch.delenv("LANGCHAIN_TRACING_V2")
    monkeypatch.delenv("LANGCHAIN_API_KEY")


def test_configure_logging_runs() -> None:
    configure_logging("DEBUG")
    configure_logging()


def test_settings_defaults_and_cors(monkeypatch: pytest.MonkeyPatch) -> None:
    s = get_settings()
    assert s.anthropic_model == DEFAULT_ANTHROPIC_MODEL == "claude-sonnet-4-5-20250929"
    assert not s.llm_enabled and s.vector_store == "in_memory"
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, http://b.com,")
    get_settings.cache_clear()
    assert get_settings().cors_origin_list == ["http://a.com", "http://b.com"]
