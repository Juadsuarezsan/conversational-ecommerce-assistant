"""Session cache: in-memory TTL and Redis (fake client)."""

from __future__ import annotations

import json
from typing import Any

from src.api.schemas import RetrievedProduct
from src.session.cache import (
    MAX_HISTORY_MESSAGES,
    InMemorySessionCache,
    RedisSessionCache,
    SessionState,
    append_turn,
    build_redis_cache,
)


def _state() -> SessionState:
    p = RetrievedProduct(product_id=1, product_name="Heinz", aisle="a", department="d", score=1.0)
    return SessionState(history=[{"role": "user", "content": "hi"}], last_products=[p], turn=1)


async def test_in_memory_cache_ttl() -> None:
    clock = {"t": 0.0}
    cache = InMemorySessionCache(ttl_seconds=10, clock=lambda: clock["t"])
    assert (await cache.get("s")).turn == 0
    await cache.set("s", _state())
    got = await cache.get("s")
    assert got.turn == 1 and got.last_products[0].product_id == 1
    got.history.append({"role": "x", "content": "y"})  # deep copy: store untouched
    assert len((await cache.get("s")).history) == 1
    clock["t"] = 11.0
    assert (await cache.get("s")).turn == 0 and len(cache) == 0
    await cache.set("s", _state())
    await cache.delete("s")
    assert len(cache) == 0


def test_append_turn_caps_history() -> None:
    s = SessionState()
    for i in range(30):
        s = append_turn(s, f"u{i}", f"a{i}")
    assert len(s.history) == MAX_HISTORY_MESSAGES and s.turn == 30
    assert s.history[-1] == {"role": "assistant", "content": "a29"}


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, int | None]] = {}

    async def get(self, key: str) -> bytes | None:
        v = self.store.get(key)
        return v[0] if v else None

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = (value.encode(), ex)

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


async def test_redis_cache_roundtrip() -> None:
    client = _FakeRedis()
    cache = RedisSessionCache(client, ttl_seconds=42, prefix="t")
    assert (await cache.get("s")).turn == 0
    await cache.set("s", _state())
    raw, ex = client.store["t:s"]
    assert ex == 42 and json.loads(raw)["turn"] == 1
    got = await cache.get("s")
    assert got.last_products[0].product_name == "Heinz"
    await cache.delete("s")
    assert "t:s" not in client.store


def test_build_redis_cache(mocker: Any) -> None:
    from_url = mocker.patch("redis.asyncio.from_url", return_value=object())
    cache = build_redis_cache("redis://h:6379/0", ttl_seconds=5, timeout=2.0)
    assert cache.ttl == 5
    from_url.assert_called_once()
