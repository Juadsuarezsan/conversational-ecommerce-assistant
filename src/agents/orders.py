"""Order lookup for the ``order_status`` intent.

The API has no order-placement flow yet, so :class:`InMemoryOrderStore` is seeded
with a handful of deterministic demo orders. The protocol is what a Postgres
implementation over the ``orders`` table would satisfy.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel

_ORDER_ID_RE = re.compile(r"#?\b(\d{4,8})\b")


class Order(BaseModel):
    """A past order."""

    order_id: str
    user_id: str
    total_usd: float
    status: str
    items: list[str]
    department: str | None = None


class OrderStore(Protocol):
    """Read-only access to orders."""

    async def get(self, order_id: str) -> Order | None:
        """Return the order or ``None``."""
        ...

    async def latest_for_user(self, user_id: str) -> Order | None:
        """Most recent order of a user."""
        ...


DEMO_ORDERS: tuple[Order, ...] = (
    Order(
        order_id="1001",
        user_id="demo-user",
        total_usd=23.47,
        status="shipped",
        items=["Heinz Tomato Ketchup 397g", "Whole Wheat Bread", "Large Eggs Dozen"],
        department="pantry",
    ),
    Order(
        order_id="1002",
        user_id="demo-user",
        total_usd=148.9,
        status="delivered",
        items=["Dry Dog Food Salmon 30lb", "Cat Litter Clumping 14lb"],
        department="pets",
    ),
    Order(
        order_id="1003",
        user_id="demo-user",
        total_usd=41.3,
        status="delivered",
        items=["Pinot Noir 750ml", "IPA Local 6pk"],
        department="alcohol",
    ),
)


class InMemoryOrderStore:
    """Deterministic demo orders."""

    def __init__(self, orders: tuple[Order, ...] = DEMO_ORDERS) -> None:
        self._orders = {o.order_id: o for o in orders}

    async def get(self, order_id: str) -> Order | None:
        """Look up by id (with or without ``#``)."""
        return self._orders.get(order_id.lstrip("#"))

    async def latest_for_user(self, user_id: str) -> Order | None:
        """Highest order id of the user (ids are monotonic in the demo)."""
        mine = [o for o in self._orders.values() if o.user_id == user_id]
        return max(mine, key=lambda o: o.order_id) if mine else None


def extract_order_id(message: str) -> str | None:
    """Return the first 4-8 digit number in the message, if any."""
    m = _ORDER_ID_RE.search(message)
    return m.group(1) if m else None
