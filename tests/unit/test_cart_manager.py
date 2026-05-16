"""CartManager state transitions."""
from __future__ import annotations

import pytest

from src.agents.cart_manager import CartManager
from src.api.schemas import CartItem


def _item(pid: int, qty: int = 1, price: float = 5.0) -> CartItem:
    return CartItem(product_id=pid, product_name=f"p{pid}", quantity=qty, unit_price_usd=price)


def test_empty_cart_total_is_zero() -> None:
    cm = CartManager()
    assert cm.view("s1") == []
    assert cm.total_usd("s1") == 0.0


def test_add_merges_duplicates() -> None:
    cm = CartManager()
    cm.add("s1", _item(1, qty=2))
    cm.add("s1", _item(1, qty=3))
    cart = cm.view("s1")
    assert len(cart) == 1
    assert cart[0].quantity == 5


def test_remove() -> None:
    cm = CartManager()
    cm.add("s1", _item(1))
    cm.add("s1", _item(2))
    cm.remove("s1", 1)
    assert {c.product_id for c in cm.view("s1")} == {2}


def test_update_to_zero_removes() -> None:
    cm = CartManager()
    cm.add("s1", _item(1, qty=4))
    cm.update_quantity("s1", 1, 0)
    assert cm.view("s1") == []


def test_total_usd() -> None:
    cm = CartManager()
    cm.add("s1", _item(1, qty=2, price=3.5))
    cm.add("s1", _item(2, qty=1, price=10.0))
    assert cm.total_usd("s1") == pytest.approx(17.0)


def test_session_isolation() -> None:
    cm = CartManager()
    cm.add("a", _item(1))
    cm.add("b", _item(2))
    assert {c.product_id for c in cm.view("a")} == {1}
    assert {c.product_id for c in cm.view("b")} == {2}
