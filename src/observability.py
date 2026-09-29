"""Structured logging, per-request cost accounting and LangSmith/Langfuse wiring.

* :func:`configure_logging` sets loguru's level and a JSON-friendly format.
* :func:`usage_from_response` reads token counts from a LangChain ``AIMessage`` and
  prices them with the configured per-million rates.
* :func:`configure_tracing` exports the environment variables LangChain reads to
  send traces to LangSmith when ``LANGSMITH_API_KEY`` is set. Nothing is sent
  otherwise.
* :func:`node_logger` wraps a graph node so its input and output state are logged
  with the request ``trace_id``.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from src.api.schemas import Usage
from src.config import Settings, get_settings

_STATE_KEYS_TO_LOG = ("intent", "escalated", "cart_action", "response")


def configure_logging(level: str | None = None) -> None:
    """Reset loguru sinks and log to stderr at ``level`` (defaults to ``LOG_LEVEL``)."""
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=(level or settings.log_level).upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "{extra[trace_id]} | <cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}"
        ),
        serialize=False,
    )
    logger.configure(extra={"trace_id": "-"})


def configure_tracing(settings: Settings | None = None) -> bool:
    """Enable LangSmith tracing through env vars when a key is configured.

    Returns:
        ``True`` when tracing was enabled.
    """
    s = settings or get_settings()
    if not s.langsmith_api_key:
        logger.info("LangSmith tracing disabled (no LANGSMITH_API_KEY)")
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
    logger.info("LangSmith tracing enabled for project {}", s.langsmith_project)
    return True


def price_usd(input_tokens: int, output_tokens: int, settings: Settings | None = None) -> float:
    """Price a call with the configured per-million-token rates."""
    s = settings or get_settings()
    cost = (input_tokens * s.llm_price_in_per_mtok + output_tokens * s.llm_price_out_per_mtok) / 1e6
    return round(cost, 6)


def usage_from_response(resp: Any, settings: Settings | None = None) -> Usage:
    """Extract token usage from a LangChain ``AIMessage`` (or anything with ``usage_metadata``)."""
    meta = getattr(resp, "usage_metadata", None) or {}
    if not meta:
        rm = getattr(resp, "response_metadata", None) or {}
        usage = rm.get("usage") or {}
        meta = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
    tin = int(meta.get("input_tokens", 0) or 0)
    tout = int(meta.get("output_tokens", 0) or 0)
    return Usage(
        input_tokens=tin, output_tokens=tout, cost_usd=price_usd(tin, tout, settings), llm_calls=1
    )


def _summarise(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _STATE_KEYS_TO_LOG:
        if key in state:
            value = state[key]
            out[key] = getattr(value, "intent", value)
    if "retrieved" in state:
        out["retrieved"] = len(state["retrieved"])
    if "reranked" in state:
        out["reranked"] = [p.product_id for p in state["reranked"]]
    if "cart" in state:
        out["cart"] = [(c.product_id, c.quantity) for c in state["cart"]]
    if "message" in state:
        out["message"] = str(state["message"])[:120]
    return out


def node_logger(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]]:
    """Decorate a LangGraph node (sync or async) to log input, output and duration."""

    def deco(fn: Callable[..., Any]) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        @functools.wraps(fn)
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            trace_id = state.get("trace_id", "-")
            log = logger.bind(trace_id=trace_id)
            log.debug("node={} in={}", name, _summarise(state))
            t0 = time.perf_counter()
            result = fn(state)
            if inspect.isawaitable(result):
                result = await result
            out: dict[str, Any] = dict(result or {})
            log.info(
                "node={} out={} ms={:.1f}", name, _summarise(out), (time.perf_counter() - t0) * 1000
            )
            return out

        return wrapper

    return deco
