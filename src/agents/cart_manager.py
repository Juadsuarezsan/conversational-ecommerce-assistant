"""Cart persistence behind a :class:`CartManager` protocol.

* :class:`InMemoryCartManager` — dict per session; used by tests and the offline API.
* :class:`PgCartManager` — ``sessions`` + ``cart_items`` tables through a psycopg
  async pool (see ``src/db/init.sql``). All SQL is parametrised.

Both are async so the orchestrator does not care which one is wired in.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Protocol

from src.api.schemas import CartItem


class CartManager(Protocol):
    """Stateful cart per ``session_id`` with the five operations of the spec."""

    async def view(self, session_id: str) -> list[CartItem]:
        """Current lines of the cart."""
        ...

    async def add(self, session_id: str, item: CartItem) -> list[CartItem]:
        """Add a line (quantities merge when the product is already present)."""
        ...

    async def remove(self, session_id: str, product_id: int) -> list[CartItem]:
        """Drop a product from the cart."""
        ...

    async def update_quantity(self, session_id: str, product_id: int, qty: int) -> list[CartItem]:
        """Set the quantity of a product; ``qty <= 0`` removes it."""
        ...

    async def clear(self, session_id: str) -> None:
        """Empty the cart."""
        ...

    async def total_usd(self, session_id: str) -> float:
        """Sum of ``unit_price_usd * quantity`` rounded to cents."""
        ...


def cart_total(items: Iterable[CartItem]) -> float:
    """Total of a list of cart lines in USD, rounded to cents."""
    return round(sum(c.unit_price_usd * c.quantity for c in items), 2)


class InMemoryCartManager:
    """Process-local carts. Lost on restart; fine for tests and demos."""

    def __init__(self) -> None:
        self._carts: dict[str, list[CartItem]] = defaultdict(list)

    async def view(self, session_id: str) -> list[CartItem]:
        """Return copies of the cart lines."""
        return [c.model_copy() for c in self._carts[session_id]]

    async def add(self, session_id: str, item: CartItem) -> list[CartItem]:
        """Add or merge a line."""
        cart = self._carts[session_id]
        for existing in cart:
            if existing.product_id == item.product_id:
                existing.quantity += item.quantity
                return await self.view(session_id)
        cart.append(item.model_copy())
        return await self.view(session_id)

    async def remove(self, session_id: str, product_id: int) -> list[CartItem]:
        """Remove a product."""
        self._carts[session_id] = [c for c in self._carts[session_id] if c.product_id != product_id]
        return await self.view(session_id)

    async def update_quantity(self, session_id: str, product_id: int, qty: int) -> list[CartItem]:
        """Set a quantity; zero or negative removes the line."""
        if qty <= 0:
            return await self.remove(session_id, product_id)
        for existing in self._carts[session_id]:
            if existing.product_id == product_id:
                existing.quantity = qty
        return await self.view(session_id)

    async def clear(self, session_id: str) -> None:
        """Empty the cart."""
        self._carts.pop(session_id, None)

    async def total_usd(self, session_id: str) -> float:
        """Cart total."""
        return cart_total(self._carts[session_id])

    async def add_many(self, session_id: str, items: Iterable[CartItem]) -> list[CartItem]:
        """Add several lines."""
        for it in items:
            await self.add(session_id, it)
        return await self.view(session_id)


class PgCartManager:
    """Carts stored in Postgres (``sessions`` + ``cart_items`` joined with ``products``).

    Args:
        pool: An opened ``psycopg_pool.AsyncConnectionPool`` (or any object with the
            same ``connection()`` async context manager). Injected so tests can pass
            a fake.
    """

    _SELECT = """
        SELECT ci.product_id, p.product_name, ci.quantity, p.price_usd
          FROM cart_items ci JOIN products p ON p.product_id = ci.product_id
         WHERE ci.session_id = %s
         ORDER BY ci.added_at, ci.id
    """

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    @staticmethod
    def _rows_to_items(rows: Iterable[tuple[Any, ...]]) -> list[CartItem]:
        return [
            CartItem(
                product_id=int(r[0]),
                product_name=str(r[1]),
                quantity=int(r[2]),
                unit_price_usd=float(r[3] or 0.0),
            )
            for r in rows
        ]

    async def _ensure_session(self, cur: Any, session_id: str, user_id: str = "anonymous") -> None:
        await cur.execute(
            """
            INSERT INTO sessions (session_id, user_id) VALUES (%s, %s)
            ON CONFLICT (session_id) DO UPDATE SET updated_at = NOW()
            """,
            (session_id, user_id),
        )

    async def view(self, session_id: str) -> list[CartItem]:
        """Read the cart lines."""
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(self._SELECT, (session_id,))
            rows = await cur.fetchall()
        return self._rows_to_items(rows)

    async def add(self, session_id: str, item: CartItem) -> list[CartItem]:
        """Insert or increment a line inside one transaction."""
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await self._ensure_session(cur, session_id)
            await cur.execute(
                "SELECT id FROM cart_items WHERE session_id = %s AND product_id = %s",
                (session_id, item.product_id),
            )
            row = await cur.fetchone()
            if row:
                await cur.execute(
                    "UPDATE cart_items SET quantity = quantity + %s WHERE id = %s",
                    (item.quantity, row[0]),
                )
            else:
                await cur.execute(
                    "INSERT INTO cart_items (session_id, product_id, quantity) VALUES (%s, %s, %s)",
                    (session_id, item.product_id, item.quantity),
                )
            await conn.commit()
        return await self.view(session_id)

    async def remove(self, session_id: str, product_id: int) -> list[CartItem]:
        """Delete a line."""
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM cart_items WHERE session_id = %s AND product_id = %s",
                (session_id, product_id),
            )
            await conn.commit()
        return await self.view(session_id)

    async def update_quantity(self, session_id: str, product_id: int, qty: int) -> list[CartItem]:
        """Set a quantity; zero or negative deletes the line."""
        if qty <= 0:
            return await self.remove(session_id, product_id)
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "UPDATE cart_items SET quantity = %s WHERE session_id = %s AND product_id = %s",
                (qty, session_id, product_id),
            )
            await conn.commit()
        return await self.view(session_id)

    async def clear(self, session_id: str) -> None:
        """Delete all lines of the session."""
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("DELETE FROM cart_items WHERE session_id = %s", (session_id,))
            await conn.commit()

    async def total_usd(self, session_id: str) -> float:
        """Cart total computed from the current lines."""
        return cart_total(await self.view(session_id))


async def open_pg_pool(
    dsn: str, min_size: int = 1, max_size: int = 8, timeout: float = 10.0
) -> Any:
    """Open a psycopg async pool (imported lazily so tests never need Postgres)."""
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        dsn.replace("postgresql+psycopg://", "postgresql://"),
        min_size=min_size,
        max_size=max_size,
        timeout=timeout,
        open=False,
    )
    await pool.open()
    return pool
