"""Session cache behind a :class:`SessionCache` protocol.

The cache stores, per ``session_id``:

* the conversation history (last N turns), and
* the products shown in the last product search, so follow-ups such as
  "add the second one" can be resolved.

:class:`InMemorySessionCache` is the offline default; :class:`RedisSessionCache`
uses ``redis.asyncio`` with the same TTL semantics. The Redis client is injected,
which is how tests exercise the class without a server.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.api.schemas import RetrievedProduct

MAX_HISTORY_MESSAGES = 20


class SessionState(BaseModel):
    """Everything the agent remembers about a session between turns."""

    history: list[dict[str, str]] = Field(default_factory=list)
    last_products: list[RetrievedProduct] = Field(default_factory=list)
    turn: int = 0


class SessionCache(Protocol):
    """Read/write :class:`SessionState` with a TTL."""

    async def get(self, session_id: str) -> SessionState:
        """Return the state (empty when missing or expired)."""
        ...

    async def set(self, session_id: str, state: SessionState) -> None:
        """Persist the state and refresh its TTL."""
        ...

    async def delete(self, session_id: str) -> None:
        """Forget the session."""
        ...


def append_turn(state: SessionState, user: str, assistant: str) -> SessionState:
    """Return a copy of ``state`` with one more user/assistant exchange."""
    history = [
        *state.history,
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ][-MAX_HISTORY_MESSAGES:]
    return state.model_copy(update={"history": history, "turn": state.turn + 1})


class InMemorySessionCache:
    """Dict-backed cache with lazy expiry (checked on read)."""

    def __init__(self, ttl_seconds: int = 1800, clock: Any = time.monotonic) -> None:
        self.ttl = ttl_seconds
        self._clock = clock
        self._data: dict[str, tuple[float, SessionState]] = {}

    async def get(self, session_id: str) -> SessionState:
        """Return the state or an empty one when missing/expired."""
        entry = self._data.get(session_id)
        if entry is None:
            return SessionState()
        expires_at, state = entry
        if self._clock() >= expires_at:
            self._data.pop(session_id, None)
            return SessionState()
        return state.model_copy(deep=True)

    async def set(self, session_id: str, state: SessionState) -> None:
        """Store and refresh TTL."""
        self._data[session_id] = (self._clock() + self.ttl, state.model_copy(deep=True))

    async def delete(self, session_id: str) -> None:
        """Forget the session."""
        self._data.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._data)


class RedisSessionCache:
    """Redis-backed cache; one JSON blob per session under ``prefix:session_id``.

    Args:
        client: An ``redis.asyncio.Redis``-compatible client (``get``/``set``/``delete``).
        ttl_seconds: Expiry applied on every write.
        prefix: Key namespace.
    """

    def __init__(self, client: Any, ttl_seconds: int = 1800, prefix: str = "ecom:session") -> None:
        self.client = client
        self.ttl = ttl_seconds
        self.prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    async def get(self, session_id: str) -> SessionState:
        """Load and validate the JSON blob; empty state when missing."""
        raw = await self.client.get(self._key(session_id))
        if not raw:
            return SessionState()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return SessionState.model_validate(json.loads(raw))

    async def set(self, session_id: str, state: SessionState) -> None:
        """Write the blob with ``EX ttl``."""
        await self.client.set(self._key(session_id), state.model_dump_json(), ex=self.ttl)

    async def delete(self, session_id: str) -> None:
        """Delete the key."""
        await self.client.delete(self._key(session_id))


def build_redis_cache(url: str, ttl_seconds: int, timeout: float = 5.0) -> RedisSessionCache:
    """Create a :class:`RedisSessionCache` from a URL (client imported lazily)."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(url, socket_timeout=timeout, socket_connect_timeout=timeout)
    return RedisSessionCache(client, ttl_seconds=ttl_seconds)
