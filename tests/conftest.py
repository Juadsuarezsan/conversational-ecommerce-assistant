"""Shared fixtures: fixed seeds, catalog, indexes, fake LLM clients, fake DB pools, API client."""

from __future__ import annotations

import os
import random
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.config import get_settings

SEED = 20260516


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    random.seed(SEED)


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force the deterministic offline configuration for every test."""
    for key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "COHERE_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("RERANK_BACKEND", "lexical")
    monkeypatch.setenv("VECTOR_STORE", "in_memory")
    monkeypatch.setenv("CART_BACKEND", "memory")
    monkeypatch.setenv("SESSION_BACKEND", "memory")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("RATE_LIMIT", "1000/minute")
    get_settings.cache_clear()
    from src.retrieval.embeddings import get_embedding_provider
    from src.retrieval.rerank import get_reranker

    get_embedding_provider.cache_clear()
    get_reranker.cache_clear()
    yield
    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    get_reranker.cache_clear()


@pytest.fixture(scope="session")
def catalog() -> list[dict[str, Any]]:
    from src.ingestion.synthetic_catalog import build_catalog

    return build_catalog()


@pytest.fixture(scope="session")
def bm25_index(catalog: list[dict[str, Any]]) -> Any:
    from src.retrieval.bm25 import BM25Index

    return BM25Index(catalog)


@pytest.fixture
def hash_embedder() -> Any:
    from src.retrieval.embeddings import HashEmbeddings

    return HashEmbeddings()


@pytest.fixture
async def runtime() -> AsyncIterator[Any]:
    """Offline AgentRuntime with a compiled graph."""
    from src.agents.orchestrator import build_graph
    from src.agents.runtime import build_runtime

    rt = await build_runtime()
    rt.graph = build_graph(rt)
    yield rt
    await rt.close()


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    """TestClient with the lifespan run (indexes built once per test)."""
    from src.api.main import create_app

    with TestClient(create_app()) as client:
        yield client


class FakeChat:
    """Stand-in for ``ChatAnthropic``: returns scripted replies with usage metadata."""

    def __init__(self, replies: list[str], input_tokens: int = 120, output_tokens: int = 40):
        self.replies = list(replies)
        self.calls: list[Any] = []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls.append(messages)
        if not self.replies:
            raise TimeoutError("no more scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(
            content=reply,
            usage_metadata={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
        )


@pytest.fixture
def fake_chat_factory() -> Any:
    return FakeChat


class FakeCursor:
    """Records executed SQL and serves scripted ``fetchall``/``fetchone`` results."""

    def __init__(self, results: list[list[tuple[Any, ...]]] | None = None) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.results = list(results or [])

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((" ".join(sql.split()), params))

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self.results.pop(0) if self.results else []

    async def fetchone(self) -> tuple[Any, ...] | None:
        rows = await self.fetchall()
        return rows[0] if rows else None

    async def __aenter__(self) -> FakeCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return self._cursor

    async def commit(self) -> None:
        self.commits += 1

    async def __aenter__(self) -> FakeConnection:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class FakePool:
    """Mimics ``psycopg_pool.AsyncConnectionPool.connection()``."""

    def __init__(self, results: list[list[tuple[Any, ...]]] | None = None) -> None:
        self.cursor = FakeCursor(results)
        self.conn = FakeConnection(self.cursor)
        self.closed = False

    def connection(self) -> FakeConnection:
        return self.conn

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_pool_factory() -> Any:
    return FakePool


@pytest.fixture
def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
