"""Stateful cart per session. In-memory store with Postgres sync (optional)."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.api.schemas import CartItem


class CartManager:
    """In-memory MVP. Use PgCartManager for persistence."""

    def __init__(self) -> None:
        self._carts: dict[str, list[CartItem]] = defaultdict(list)

    def view(self, session_id: str) -> list[CartItem]:
        return list(self._carts[session_id])

    def add(self, session_id: str, item: CartItem) -> list[CartItem]:
        cart = self._carts[session_id]
        for existing in cart:
            if existing.product_id == item.product_id:
                existing.quantity += item.quantity
                return self.view(session_id)
        cart.append(item)
        return self.view(session_id)

    def remove(self, session_id: str, product_id: int) -> list[CartItem]:
        self._carts[session_id] = [c for c in self._carts[session_id] if c.product_id != product_id]
        return self.view(session_id)

    def update_quantity(self, session_id: str, product_id: int, qty: int) -> list[CartItem]:
        if qty <= 0:
            return self.remove(session_id, product_id)
        for existing in self._carts[session_id]:
            if existing.product_id == product_id:
                existing.quantity = qty
        return self.view(session_id)

    def clear(self, session_id: str) -> None:
        self._carts.pop(session_id, None)

    def total_usd(self, session_id: str) -> float:
        return round(sum(c.unit_price_usd * c.quantity for c in self._carts[session_id]), 2)

    def add_many(self, session_id: str, items: Iterable[CartItem]) -> list[CartItem]:
        for it in items:
            self.add(session_id, it)
        return self.view(session_id)
