"""Structured Retriever: price / category / rating / availability filters over the catalog.

Two implementations share :class:`StructuredFilter`:

* :class:`InMemoryStructuredRetriever` filters the catalog list (tests, offline API).
* :class:`PgStructuredRetriever` compiles the same filter into a parametrised SQL
  query against the ``products`` table (Postgres). :func:`build_sql` is pure so
  it can be unit-tested without a database.

:func:`parse_filters` extracts filters from free text ("cereal under $5",
"best rated", "cheapest"). The Intent Router may also provide filters; both are
merged in the orchestrator.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.api.schemas import RetrievedProduct
from src.retrieval.bm25 import to_retrieved, tokenize

SortKey = Literal["relevance", "price_asc", "price_desc", "rating_desc", "popularity_desc"]

_PRICE_MAX_RE = re.compile(
    r"(?:under|below|less than|cheaper than|at most|max(?:imum)?|up to)\s*\$?\s*(\d+(?:\.\d+)?)"
)
_PRICE_MIN_RE = re.compile(
    r"(?:over|above|more than|at least|min(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)"
)
_RATING_RE = re.compile(r"(\d(?:\.\d)?)\s*(?:\+|stars?|or better|and up)")
_DIET_WORDS = {
    "gluten_free": ("gluten free", "gluten-free"),
    "lactose_free": ("lactose free", "lactose-free", "dairy free", "dairy-free"),
    "vegan": ("vegan", "plant based", "plant-based"),
    "organic": ("organic",),
}


class StructuredFilter(BaseModel):
    """Declarative filter + sort applied on top of (or instead of) semantic retrieval."""

    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    department: str | None = None
    aisle: str | None = None
    min_rating: float | None = Field(default=None, ge=0, le=5)
    in_stock_only: bool = False
    diet: str | None = None
    sort: SortKey = "relevance"

    def is_empty(self) -> bool:
        """True when no filter or sort is active."""
        return self == StructuredFilter()

    def merged(self, other: StructuredFilter) -> StructuredFilter:
        """Return a copy where unset fields are filled from ``other``."""
        data = self.model_dump()
        for key, value in other.model_dump().items():
            if data[key] in (None, False, "relevance"):
                data[key] = value
        return StructuredFilter(**data)


def parse_filters(message: str) -> StructuredFilter:
    """Extract price / rating / diet / sort hints from a message with regexes.

    Args:
        message: Raw user text.

    Returns:
        A :class:`StructuredFilter`; empty when nothing was recognised.
    """
    m = message.lower()
    f = StructuredFilter()
    if pm := _PRICE_MAX_RE.search(m):
        f.price_max = float(pm.group(1))
    if pn := _PRICE_MIN_RE.search(m):
        f.price_min = float(pn.group(1))
    if rt := _RATING_RE.search(m):
        rating = float(rt.group(1))
        if rating <= 5:
            f.min_rating = rating
    if any(w in m for w in ("cheapest", "cheaper", "lowest price", "budget", "affordable")):
        f.sort = "price_asc"
    elif any(w in m for w in ("premium", "most expensive", "luxury")):
        f.sort = "price_desc"
    elif any(w in m for w in ("best rated", "top rated", "highest rated", "best reviewed")):
        f.sort = "rating_desc"
    elif any(w in m for w in ("most popular", "bestseller", "best seller", "popular")):
        f.sort = "popularity_desc"
    if "in stock" in m or "available" in m:
        f.in_stock_only = True
    for diet, words in _DIET_WORDS.items():
        if any(w in m for w in words):
            f.diet = diet
            break
    return f


def filters_from_router(extracted: dict[str, str | float | int | bool]) -> StructuredFilter:
    """Map the Intent Router's ``extracted_filters`` dict into a :class:`StructuredFilter`."""
    f = StructuredFilter()
    price_max = extracted.get("price_max")
    if isinstance(price_max, int | float) and not isinstance(price_max, bool):
        f.price_max = float(price_max)
    price_min = extracted.get("price_min")
    if isinstance(price_min, int | float) and not isinstance(price_min, bool):
        f.price_min = float(price_min)
    category = extracted.get("category")
    if isinstance(category, str) and category:
        f.department = category.lower()
    diet = extracted.get("diet")
    if isinstance(diet, str) and diet:
        f.diet = diet.lower().replace("-", "_").replace(" ", "_")
    return f


def _matches_diet(p: dict[str, Any], diet: str) -> bool:
    name = str(p.get("product_name", "")).lower()
    words = _DIET_WORDS.get(diet, (diet.replace("_", " "),))
    return any(w in name for w in words)


def _passes(p: dict[str, Any], f: StructuredFilter) -> bool:
    price = p.get("price_usd")
    if f.price_max is not None and (price is None or float(price) > f.price_max):
        return False
    if f.price_min is not None and (price is None or float(price) < f.price_min):
        return False
    if f.department and str(p.get("department", "")).lower() != f.department.lower():
        return False
    if f.aisle and str(p.get("aisle", "")).lower() != f.aisle.lower():
        return False
    if f.min_rating is not None:
        rating = p.get("avg_rating")
        if rating is None or float(rating) < f.min_rating:
            return False
    if f.in_stock_only and not bool(p.get("in_stock", True)):
        return False
    return not (f.diet and not _matches_diet(p, f.diet))


def _sort_key(f: StructuredFilter) -> Any:
    if f.sort == "price_asc":
        return lambda p: (p.get("price_usd") is None, float(p.get("price_usd") or 0.0))
    if f.sort == "price_desc":
        return lambda p: -float(p.get("price_usd") or 0.0)
    if f.sort == "rating_desc":
        return lambda p: (-float(p.get("avg_rating") or 0.0), -int(p.get("rating_count") or 0))
    if f.sort == "popularity_desc":
        return lambda p: -int(p.get("rating_count") or 0)
    return None


_FILTER_WORDS = frozenset(
    {
        "cheapest",
        "cheaper",
        "cheap",
        "best",
        "top",
        "rated",
        "rating",
        "under",
        "over",
        "below",
        "less",
        "than",
        "most",
        "popular",
        "version",
        "of",
        "premium",
        "budget",
        "affordable",
        "price",
        "priced",
        "in",
        "stock",
        "available",
        "the",
        "a",
        "an",
        "for",
        "me",
        "show",
        "find",
        "some",
        "any",
        "with",
        "and",
        "or",
        "good",
        "great",
        "highest",
        "lowest",
        "stars",
        "star",
        "reviewed",
        "bestseller",
        "expensive",
        "luxury",
    }
)


def lexically_related(query: str, candidates: Sequence[RetrievedProduct]) -> list[RetrievedProduct]:
    """Keep candidates sharing at least one descriptive query token with their text.

    Used before a price/rating sort so "cheapest coffee" sorts coffee products and
    not the whole fused list. Filter words ("cheapest", "under", ...) are ignored.
    """
    q_tokens = {t for t in tokenize(query) if t not in _FILTER_WORDS and not t.isdigit()}
    if not q_tokens:
        return list(candidates)
    kept = [
        c
        for c in candidates
        if q_tokens & set(tokenize(f"{c.product_name} {c.aisle} {c.department}"))
    ]
    return kept


def apply_filter(
    candidates: Sequence[RetrievedProduct], f: StructuredFilter
) -> list[RetrievedProduct]:
    """Filter and optionally re-sort already retrieved candidates.

    Semantic order is preserved for ``sort="relevance"``; otherwise the candidates
    are re-sorted by the requested key and re-tagged ``source="structured"``.
    """
    kept = [c for c in candidates if _passes(c.model_dump(), f)]
    key = _sort_key(f)
    if key is None:
        return kept
    ordered = sorted(kept, key=lambda c: key(c.model_dump()))
    return [
        RetrievedProduct(
            **{**c.model_dump(), "source": "structured", "score": float(len(ordered) - i)}
        )
        for i, c in enumerate(ordered)
    ]


class InMemoryStructuredRetriever:
    """Filter the whole catalog list (no text relevance)."""

    name = "in_memory"

    def __init__(self, products: Sequence[dict[str, Any]]) -> None:
        self.products = list(products)

    async def search(self, f: StructuredFilter, k: int = 20) -> list[RetrievedProduct]:
        """Return up to ``k`` catalog rows passing ``f``, sorted by ``f.sort``."""
        kept = [p for p in self.products if _passes(p, f)]
        key = _sort_key(f)
        if key is not None:
            kept.sort(key=key)
        return [to_retrieved(p, float(len(kept) - i), "structured") for i, p in enumerate(kept[:k])]


_SORT_SQL: dict[str, str] = {
    "relevance": "product_id ASC",
    "price_asc": "price_usd ASC NULLS LAST, product_id ASC",
    "price_desc": "price_usd DESC NULLS LAST, product_id ASC",
    "rating_desc": "avg_rating DESC NULLS LAST, rating_count DESC, product_id ASC",
    "popularity_desc": "rating_count DESC, product_id ASC",
}


def build_sql(f: StructuredFilter, k: int = 20) -> tuple[str, list[Any]]:
    """Compile a :class:`StructuredFilter` into a parametrised ``SELECT``.

    Returns:
        ``(sql, params)`` using ``%s`` placeholders (psycopg style). Column names
        are fixed; user input only ever reaches the parameter list.
    """
    where: list[str] = []
    params: list[Any] = []
    if f.price_max is not None:
        where.append("price_usd <= %s")
        params.append(f.price_max)
    if f.price_min is not None:
        where.append("price_usd >= %s")
        params.append(f.price_min)
    if f.department:
        where.append("LOWER(department) = %s")
        params.append(f.department.lower())
    if f.aisle:
        where.append("LOWER(aisle) = %s")
        params.append(f.aisle.lower())
    if f.min_rating is not None:
        where.append("avg_rating >= %s")
        params.append(f.min_rating)
    if f.in_stock_only:
        where.append("in_stock = TRUE")
    if f.diet:
        words = _DIET_WORDS.get(f.diet, (f.diet.replace("_", " "),))
        where.append("(" + " OR ".join("product_name ILIKE %s" for _ in words) + ")")
        params.extend(f"%{w}%" for w in words)
    sql = (
        "SELECT product_id, product_name, aisle, department, price_usd, avg_rating, "
        "rating_count, in_stock FROM products"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {_SORT_SQL[f.sort]} LIMIT %s"
    params.append(k)
    return sql, params


class PgStructuredRetriever:
    """Run :func:`build_sql` against Postgres through an injected async pool."""

    name = "postgres"

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def search(self, f: StructuredFilter, k: int = 20) -> list[RetrievedProduct]:
        """Execute the compiled SQL and map rows to :class:`RetrievedProduct`."""
        sql, params = build_sql(f, k)
        async with self.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        out: list[RetrievedProduct] = []
        for i, r in enumerate(rows):
            out.append(
                to_retrieved(
                    {
                        "product_id": r[0],
                        "product_name": r[1],
                        "aisle": r[2],
                        "department": r[3],
                        "price_usd": float(r[4]) if r[4] is not None else None,
                        "avg_rating": float(r[5]) if r[5] is not None else None,
                        "rating_count": r[6],
                        "in_stock": r[7],
                    },
                    float(len(rows) - i),
                    "structured",
                )
            )
        return out
