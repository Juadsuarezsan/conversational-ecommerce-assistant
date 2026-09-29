"""CartManager implementations: in-memory and Postgres (fake pool)."""

from __future__ import annotations

from typing import Any

import pytest

from src.agents.cart_manager import InMemoryCartManager, PgCartManager, cart_total
from src.api.schemas import CartItem


def _item(pid: int, qty: int = 1, price: float = 5.0) -> CartItem:
    return CartItem(product_id=pid, product_name=f"p{pid}", quantity=qty, unit_price_usd=price)


async def test_empty_cart_total_is_zero() -> None:
    cm = InMemoryCartManager()
    assert await cm.view("s1") == []
    assert await cm.total_usd("s1") == 0.0


async def test_add_merges_duplicates() -> None:
    cm = InMemoryCartManager()
    await cm.add("s1", _item(1, qty=2))
    cart = await cm.add("s1", _item(1, qty=3))
    assert len(cart) == 1 and cart[0].quantity == 5


async def test_remove_and_update() -> None:
    cm = InMemoryCartManager()
    await cm.add("s1", _item(1, qty=4))
    await cm.add("s1", _item(2))
    await cm.remove("s1", 1)
    assert {c.product_id for c in await cm.view("s1")} == {2}
    await cm.update_quantity("s1", 2, 7)
    assert (await cm.view("s1"))[0].quantity == 7
    await cm.update_quantity("s1", 2, 0)
    assert await cm.view("s1") == []


async def test_clear_and_total() -> None:
    cm = InMemoryCartManager()
    await cm.add_many("s1", [_item(1, qty=2, price=3.5), _item(2, qty=1, price=10.0)])
    assert await cm.total_usd("s1") == pytest.approx(17.0)
    await cm.clear("s1")
    assert await cm.view("s1") == []


async def test_session_isolation_and_copies() -> None:
    cm = InMemoryCartManager()
    await cm.add("a", _item(1))
    await cm.add("b", _item(2))
    assert {c.product_id for c in await cm.view("a")} == {1}
    view = await cm.view("b")
    view[0].quantity = 99  # mutating the copy must not touch the store
    assert (await cm.view("b"))[0].quantity == 1


def test_cart_total_rounds() -> None:
    assert cart_total([_item(1, qty=3, price=0.1)]) == 0.3


async def test_pg_cart_manager_add_insert_then_increment(fake_pool_factory: Any) -> None:
    pool = fake_pool_factory(
        results=[[], [(1, "Ketchup", 2, 4.29)], [(7,)], [(1, "Ketchup", 5, 4.29)]]
    )
    pg = PgCartManager(pool)
    cart = await pg.add("s1", _item(1, qty=2))
    assert cart[0].quantity == 2 and pool.conn.commits == 1
    sql = " ".join(s for s, _ in pool.cursor.executed)
    assert "INSERT INTO sessions" in sql and "INSERT INTO cart_items" in sql
    cart = await pg.add("s1", _item(1, qty=3))
    assert cart[0].quantity == 5
    assert any("quantity = quantity + %s" in s for s, _ in pool.cursor.executed)


async def test_pg_cart_manager_remove_update_clear_total(fake_pool_factory: Any) -> None:
    pool = fake_pool_factory(results=[[], [(2, "Bread", 3, 2.0)], [], [(2, "Bread", 3, 2.0)]])
    pg = PgCartManager(pool)
    assert await pg.remove("s1", 1) == []
    cart = await pg.update_quantity("s1", 2, 3)
    assert cart[0].quantity == 3
    assert await pg.update_quantity("s1", 2, 0) == []  # delegates to remove
    await pg.clear("s1")
    assert any(
        s.startswith("DELETE FROM cart_items WHERE session_id = %s")
        for s, _ in pool.cursor.executed
    )
    assert await pg.total_usd("s1") == 6.0
    assert all(p is None or isinstance(p, tuple) for _, p in pool.cursor.executed)
