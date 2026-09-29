"""Cart operation parsing, product reference resolution and execution."""

from __future__ import annotations

from typing import Any

from src.agents.cart_manager import InMemoryCartManager
from src.agents.cart_ops import (
    detect_action,
    execute_operation,
    parse_operation,
    parse_ordinal,
    parse_quantity,
    pick_superlative,
    resolve_product,
)
from src.api.schemas import CartItem, RetrievedProduct


def _rp(
    pid: int, name: str, price: float = 1.0, rating: float = 4.0, aisle: str = "x"
) -> RetrievedProduct:
    return RetrievedProduct(
        product_id=pid,
        product_name=name,
        aisle=aisle,
        department="d",
        price_usd=price,
        avg_rating=rating,
        rating_count=pid * 10,
        score=1.0,
        source="rerank",
    )


LAST = [
    _rp(6, "Lactose-Free 2% Milk 1 Gallon", 5.49),
    _rp(7, "Almond Milk Unsweetened 64oz", 3.49, 4.3),
    _rp(8, "Lactose-Free Whole Milk 64oz", 4.29, 4.8),
]


def test_detect_action_variants() -> None:
    assert detect_action("add 2 to my cart") == "add"
    assert detect_action("remove the milk") == "remove"
    assert detect_action("change the quantity to 3") == "update_quantity"
    assert detect_action("make that 3") == "update_quantity"
    assert detect_action("empty my cart") == "clear"
    assert detect_action("what's in my cart") == "view"
    assert detect_action("swap that for the squeeze bottle") == "add"


def test_parse_quantity() -> None:
    assert parse_quantity("add 2 of the second one", "add") == 2
    assert parse_quantity("add three please", "add") == 3
    assert parse_quantity("add the second one", "add") == 1
    assert parse_quantity("change quantity to 4", "update_quantity") == 4
    assert parse_quantity("make that 3", "update_quantity") == 3
    assert parse_quantity("add [pid:12]", "add") == 1


def test_parse_ordinal() -> None:
    assert parse_ordinal("add the second one") == 2
    assert parse_ordinal("the last one") == -1
    assert parse_ordinal("add milk") is None


def test_resolve_by_ordinal_pid_and_name(bm25_index: Any) -> None:
    assert resolve_product("add the second one", LAST, [], bm25_index).product_id == 7
    assert resolve_product("the last one", LAST, [], bm25_index).product_id == 8
    assert resolve_product("add [pid:8]", LAST, [], bm25_index).product_id == 8
    assert resolve_product("add product 1", [], [], bm25_index).product_id == 1
    assert resolve_product("add the almond milk", LAST, [], bm25_index).product_id == 7
    # catalog fallback when the product was never shown
    assert resolve_product("add ketchup", [], [], bm25_index).product_id in (1, 2)
    assert resolve_product("add the flux capacitor", [], [], bm25_index) is None


def test_resolve_prefers_cart_for_removal(bm25_index: Any) -> None:
    cart = [
        CartItem(product_id=32, product_name="Lactose-Free Milk", quantity=1, unit_price_usd=22.8)
    ]
    # "almond milk" must not resolve to the lactose-free line just because both say milk
    target = resolve_product("remove the almond milk", LAST, cart, bm25_index, prefer_cart=True)
    assert target.product_id == 7
    target = resolve_product(
        "remove the lactose free milk", LAST, cart, bm25_index, prefer_cart=True
    )
    assert target.product_id == 32
    # pronoun with a single cart line
    assert resolve_product("remove them", [], cart, bm25_index, prefer_cart=True).product_id == 32
    assert resolve_product("add it", LAST, [], bm25_index).product_id == 6
    assert resolve_product("add it", [], [], bm25_index) is None


def test_resolve_matches_aisle_of_cart_item(bm25_index: Any) -> None:
    cart = [
        CartItem(product_id=210, product_name="Pinot Noir 750ml", quantity=2, unit_price_usd=4.35)
    ]
    assert (
        resolve_product("refund the wine", [], cart, bm25_index, prefer_cart=True).product_id == 210
    )


def test_superlatives() -> None:
    assert pick_superlative("add the cheapest one", LAST).product_id == 7
    assert pick_superlative("the best rated", LAST).product_id == 8
    assert pick_superlative("the most expensive", LAST).product_id == 6
    assert pick_superlative("the most popular", LAST).product_id == 8
    assert pick_superlative("add milk", LAST) is None


def test_parse_operation_shapes(bm25_index: Any) -> None:
    op = parse_operation("add 2 of the second one to my cart", LAST, [], bm25_index)
    assert (op.action, op.product_id, op.quantity) == ("add", 7, 2)
    op = parse_operation("what's in my cart", LAST, [], bm25_index)
    assert op.action == "view" and op.product_id is None
    op = parse_operation("add the flux capacitor", [], [], bm25_index)
    assert op.action == "add" and op.product_id is None


async def test_execute_add_remove_update_clear(bm25_index: Any) -> None:
    cm = InMemoryCartManager()
    op = parse_operation("add 2 of the second one", LAST, [], bm25_index)
    res = await execute_operation(
        op, session_id="s", cart_mgr=cm, last_products=LAST, cart=[], bm25=bm25_index
    )
    assert res.ok and res.cart[0].product_id == 7 and res.cart[0].quantity == 2
    cart = await cm.view("s")
    op = parse_operation("make that 5", LAST, cart, bm25_index)
    res = await execute_operation(
        op, session_id="s", cart_mgr=cm, last_products=LAST, cart=cart, bm25=bm25_index
    )
    assert res.ok and res.cart[0].quantity == 5
    op = parse_operation("remove the ketchup", LAST, cart, bm25_index)
    res = await execute_operation(
        op, session_id="s", cart_mgr=cm, last_products=LAST, cart=cart, bm25=bm25_index
    )
    assert not res.ok and "not in your cart" in res.message
    op = parse_operation("remove the almond milk", LAST, cart, bm25_index)
    res = await execute_operation(
        op, session_id="s", cart_mgr=cm, last_products=LAST, cart=cart, bm25=bm25_index
    )
    assert res.ok and res.cart == []
    op = parse_operation("add ketchup", [], [], bm25_index)  # resolved through the catalog
    res = await execute_operation(
        op, session_id="s", cart_mgr=cm, last_products=[], cart=[], bm25=bm25_index
    )
    assert res.ok and res.cart[0].product_id in (1, 2) and res.cart[0].unit_price_usd > 0
    op = parse_operation("empty my cart", [], [], bm25_index)
    res = await execute_operation(
        op, session_id="s", cart_mgr=cm, last_products=[], cart=[], bm25=bm25_index
    )
    assert res.ok and res.cart == []


async def test_execute_unresolved_reference_is_not_ok(bm25_index: Any) -> None:
    cm = InMemoryCartManager()
    op = parse_operation("add the flux capacitor", [], [], bm25_index)
    res = await execute_operation(
        op, session_id="s", cart_mgr=cm, last_products=[], cart=[], bm25=bm25_index
    )
    assert not res.ok and "couldn't tell" in res.message
    op = parse_operation("what's in my cart", [], [], bm25_index)
    res = await execute_operation(op, session_id="s", cart_mgr=cm, last_products=[], cart=[])
    assert res.ok and res.cart == []
