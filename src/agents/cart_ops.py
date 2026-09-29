"""Turn a ``cart_op`` message into a concrete :class:`CartOperation` and execute it.

The parser is deterministic (regex + ordinal words) so it works without an LLM and
is fully unit-testable. Product references are resolved, in order, against:

1. an explicit ``[pid:N]`` / ``product 12`` mention,
2. an ordinal ("the second one") over the products shown in the previous turn,
3. a fuzzy name match ("the ketchup") over the previous results, then the cart,
   then the whole catalog through BM25.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.agents.cart_manager import CartManager
from src.api.schemas import CartAction, CartItem, CartOperation, RetrievedProduct
from src.retrieval.bm25 import BM25Index, to_retrieved, tokenize

_ORDINALS: dict[str, int] = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "last": -1,
}
_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "dozen": 12,
}
_PID_RE = re.compile(r"(?:\[pid:|pid\s*|product\s*(?:id\s*)?#?)(\d+)\]?", re.IGNORECASE)
_QTY_RE = re.compile(
    r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|dozen)\b(?!\s*(?:st|nd|rd|th)\b)"
)
_SET_QTY_RE = re.compile(
    r"(?:change|set|update|make)\b.*?\b(?:to|=|it|that|them)\s*"
    r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
)
_SUPERLATIVES: dict[str, str] = {
    "cheapest": "price_asc",
    "cheaper": "price_asc",
    "cheap": "price_asc",
    "most expensive": "price_desc",
    "priciest": "price_desc",
    "best rated": "rating_desc",
    "top rated": "rating_desc",
    "highest rated": "rating_desc",
    "best": "rating_desc",
    "most popular": "popularity_desc",
}
_STOPWORDS = {
    "add",
    "remove",
    "delete",
    "drop",
    "take",
    "out",
    "put",
    "please",
    "the",
    "a",
    "an",
    "to",
    "my",
    "cart",
    "basket",
    "from",
    "of",
    "one",
    "ones",
    "that",
    "this",
    "it",
    "them",
    "and",
    "more",
    "in",
    "into",
    "some",
    "for",
    "me",
    "can",
    "you",
    "i",
    "want",
    "need",
    "with",
    "swap",
    "replace",
    "instead",
    "change",
    "set",
    "update",
    "quantity",
    "qty",
    "return",
    "refund",
    "bought",
    "ordered",
    "yesterday",
    "last",
    "week",
    "order",
    "back",
    "money",
    "cheapest",
    "cheaper",
    "cheap",
    "best",
    "top",
    "rated",
    "highest",
    "most",
    "expensive",
    "priciest",
    "popular",
    "too",
    "also",
    "actually",
    "those",
    "these",
    "all",
    "cans",
    "bottles",
    "packs",
    "make",
    "now",
}


@dataclass
class CartOpResult:
    """Outcome of executing a cart operation."""

    operation: CartOperation
    cart: list[CartItem]
    message: str
    ok: bool


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS.get(token, 1)


def detect_action(message: str) -> CartAction:
    """Classify the cart verb: view / add / remove / update_quantity / clear."""
    m = message.lower()
    if any(
        w in m
        for w in (
            "empty my cart",
            "clear my cart",
            "clear the cart",
            "empty the cart",
            "clear cart",
            "start over",
        )
    ):
        return "clear"
    if _SET_QTY_RE.search(m) or "quantity" in m or "qty" in m:
        return "update_quantity"
    if any(w in m for w in ("remove", "delete", "take out", "drop", "get rid of", "without")):
        return "remove"
    if any(
        w in m
        for w in (
            "add",
            "put",
            "throw in",
            "include",
            "i'll take",
            "i will take",
            "swap",
            "replace",
        )
    ):
        return "add"
    if any(
        w in m
        for w in ("what's in", "whats in", "show", "view", "see", "my cart", "checkout", "total")
    ):
        return "view"
    return "view"


def parse_quantity(message: str, action: str) -> int:
    """Extract a quantity (default 1) appropriate for the action."""
    m = message.lower()
    if action == "update_quantity":
        if sm := _SET_QTY_RE.search(m):
            return _to_int(sm.group(1))
    # Ignore numbers that are part of a pid reference or an ordinal.
    cleaned = _PID_RE.sub(" ", m)
    for tok in _QTY_RE.findall(cleaned):
        val = _to_int(tok)
        if 0 < val <= 99:
            return val
    return 1


def parse_ordinal(message: str) -> int | None:
    """Return the 1-based ordinal mentioned ("second one" → 2, "last" → -1)."""
    m = message.lower()
    for word, idx in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", m):
            return idx
    return None


def pick_superlative(
    message: str, last_products: Sequence[RetrievedProduct]
) -> RetrievedProduct | None:
    """Resolve "the cheapest one" / "the best rated" over the previous results."""
    m = message.lower()
    for word, sort in _SUPERLATIVES.items():
        if re.search(rf"\b{re.escape(word)}\b", m) and last_products:
            if sort == "price_asc":
                return min(last_products, key=lambda p: (p.price_usd is None, p.price_usd or 0.0))
            if sort == "price_desc":
                return max(last_products, key=lambda p: p.price_usd or 0.0)
            if sort == "rating_desc":
                return max(last_products, key=lambda p: (p.avg_rating or 0.0, p.rating_count))
            return max(last_products, key=lambda p: p.rating_count)
    return None


def _name_tokens(message: str) -> set[str]:
    cleaned = _PID_RE.sub(" ", message.lower())
    return {
        t
        for t in tokenize(cleaned)
        if t not in _STOPWORDS and t not in _ORDINALS and not t.isdigit()
    }


def _candidate_text(item: RetrievedProduct | CartItem, bm25: BM25Index | None) -> str:
    """Name plus aisle/department so "the wine" can match a Pinot Noir in the cart."""
    if isinstance(item, RetrievedProduct):
        return f"{item.product_name} {item.aisle} {item.department}"
    row = bm25.get(item.product_id) if bm25 is not None else None
    if row is None:
        return item.product_name
    return f"{item.product_name} {row.get('aisle', '')} {row.get('department', '')}"


def _best_name_match(tokens: set[str], candidates: Sequence[tuple[int, str, int]]) -> int | None:
    """Return the candidate id that covers most of the user's descriptive tokens.

    Args:
        tokens: Descriptive tokens from the message (stop words removed).
        candidates: ``(product_id, product_name, pool_rank)`` tuples; lower pool rank
            wins ties (e.g. cart before catalog for a removal).

    A candidate must cover at least half of the user's tokens, so "almond milk"
    never resolves to "Lactose-Free Milk" when "Almond Milk" is also a candidate.
    """
    best_id: int | None = None
    best_key: tuple[float, float, float] | None = None
    for pid, name, pool_rank in candidates:
        name_toks = set(tokenize(name))
        overlap = len(tokens & name_toks)
        if overlap == 0:
            continue
        user_cov = overlap / len(tokens)
        if user_cov < 0.5:
            continue
        key = (user_cov, -float(pool_rank), overlap / len(name_toks))
        if best_key is None or key > best_key:
            best_key, best_id = key, pid
    return best_id


def resolve_product(
    message: str,
    last_products: Sequence[RetrievedProduct],
    cart: Sequence[CartItem],
    bm25: BM25Index | None,
    *,
    prefer_cart: bool = False,
) -> RetrievedProduct | CartItem | None:
    """Resolve the product the user is talking about (see module docstring).

    Args:
        message: User text.
        last_products: Products shown in the previous search turn.
        cart: Current cart lines.
        bm25: Catalog index for name lookups (optional).
        prefer_cart: Break ties in favour of cart lines (remove / update).
    """
    if pm := _PID_RE.search(message):
        pid = int(pm.group(1))
        for p in last_products:
            if p.product_id == pid:
                return p
        for c in cart:
            if c.product_id == pid:
                return c
        if bm25 is not None and (row := bm25.get(pid)) is not None:
            hits = bm25.search(str(row["product_name"]), k=1)
            if hits and hits[0].product_id == pid:
                return hits[0]
    ordinal = parse_ordinal(message)
    if ordinal is not None and last_products:
        idx = ordinal - 1 if ordinal > 0 else len(last_products) - 1
        if 0 <= idx < len(last_products):
            return last_products[idx]
    superlative = pick_superlative(message, last_products)
    if superlative is not None:
        return superlative
    tokens = _name_tokens(message)
    if not tokens:
        # Pronoun only ("add it", "remove them"): the single cart line, else the top result.
        if prefer_cart and len(cart) == 1:
            return cart[0]
        if last_products:
            return last_products[0]
        return None
    pools: list[Sequence[RetrievedProduct | CartItem]] = [last_products, cart]
    if prefer_cart:
        pools.reverse()
    catalog_hits: Sequence[RetrievedProduct | CartItem] = (
        bm25.search(" ".join(sorted(tokens)), k=3) if bm25 is not None else []
    )
    pools.append(catalog_hits)
    candidates = [
        (p.product_id, _candidate_text(p, bm25), rank)
        for rank, pool in enumerate(pools)
        for p in pool
    ]
    best = _best_name_match(tokens, candidates)
    if best is None:
        return None
    for pool in pools:
        for item in pool:
            if item.product_id == best:
                return item
    return None


def parse_operation(
    message: str,
    last_products: Sequence[RetrievedProduct],
    cart: Sequence[CartItem],
    bm25: BM25Index | None = None,
) -> CartOperation:
    """Build a :class:`CartOperation` from free text plus session context."""
    action = detect_action(message)
    if action in ("view", "clear"):
        return CartOperation(action=action, reference=message.strip())
    target = resolve_product(
        message, last_products, cart, bm25, prefer_cart=action in ("remove", "update_quantity")
    )
    qty = parse_quantity(message, action)
    return CartOperation(
        action=action,
        product_id=target.product_id if target else None,
        quantity=qty,
        reference=target.product_name if target else message.strip(),
    )


def _to_cart_item(target: RetrievedProduct | CartItem, qty: int) -> CartItem:
    price = (
        target.unit_price_usd if isinstance(target, CartItem) else float(target.price_usd or 0.0)
    )
    return CartItem(
        product_id=target.product_id,
        product_name=target.product_name,
        quantity=qty,
        unit_price_usd=price,
    )


async def execute_operation(
    op: CartOperation,
    *,
    session_id: str,
    cart_mgr: CartManager,
    last_products: Sequence[RetrievedProduct],
    cart: Sequence[CartItem],
    bm25: BM25Index | None = None,
) -> CartOpResult:
    """Apply ``op`` through ``cart_mgr`` and return the new cart with a short message."""
    if op.action == "view":
        items = await cart_mgr.view(session_id)
        return CartOpResult(op, items, "Here is your cart.", True)
    if op.action == "clear":
        await cart_mgr.clear(session_id)
        return CartOpResult(op, [], "Your cart is now empty.", True)
    if op.product_id is None:
        items = await cart_mgr.view(session_id)
        return CartOpResult(
            op,
            items,
            "I couldn't tell which product you mean. Search for it first or name it explicitly.",
            False,
        )
    target: RetrievedProduct | CartItem | None = next(
        (p for p in last_products if p.product_id == op.product_id), None
    ) or next((c for c in cart if c.product_id == op.product_id), None)
    if target is None and bm25 is not None and (row := bm25.get(op.product_id)) is not None:
        target = to_retrieved(row, 0.0, "structured")
    if op.action == "remove":
        if not any(c.product_id == op.product_id for c in cart):
            items = await cart_mgr.view(session_id)
            return CartOpResult(op, items, f"{op.reference} is not in your cart.", False)
        items = await cart_mgr.remove(session_id, op.product_id)
        return CartOpResult(op, items, f"Removed {op.reference} from your cart.", True)
    if op.action == "update_quantity":
        items = await cart_mgr.update_quantity(session_id, op.product_id, op.quantity)
        return CartOpResult(op, items, f"Set {op.reference} to {op.quantity}.", True)
    if target is None:
        items = await cart_mgr.view(session_id)
        return CartOpResult(op, items, "That product is not available right now.", False)
    items = await cart_mgr.add(session_id, _to_cart_item(target, max(op.quantity, 1)))
    return CartOpResult(
        op, items, f"Added {max(op.quantity, 1)} x {target.product_name} to your cart.", True
    )
