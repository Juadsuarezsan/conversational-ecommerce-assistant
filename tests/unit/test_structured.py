"""Structured retriever: filter parsing, in-memory filtering, SQL compilation, Postgres (fake pool)."""

from __future__ import annotations

from typing import Any

from src.api.schemas import RetrievedProduct
from src.retrieval.structured import (
    InMemoryStructuredRetriever,
    PgStructuredRetriever,
    StructuredFilter,
    apply_filter,
    build_sql,
    filters_from_router,
    lexically_related,
    parse_filters,
)


def test_parse_filters_price_rating_sort_diet() -> None:
    f = parse_filters("top rated cereals under $5")
    assert f.price_max == 5.0 and f.sort == "rating_desc"
    f = parse_filters("gluten free bread over $3 with 4 stars, in stock")
    assert (
        f.price_min == 3.0 and f.min_rating == 4.0 and f.diet == "gluten_free" and f.in_stock_only
    )
    assert parse_filters("cheapest coffee").sort == "price_asc"
    assert parse_filters("premium olive oil").sort == "price_desc"
    assert parse_filters("most popular eggs").sort == "popularity_desc"
    assert parse_filters("bananas").is_empty()


def test_filters_from_router_and_merge() -> None:
    f = filters_from_router(
        {"price_max": 10, "category": "Snacks", "diet": "Lactose-Free", "price_min": True}
    )
    assert (
        f.price_max == 10.0
        and f.department == "snacks"
        and f.diet == "lactose_free"
        and f.price_min is None
    )
    merged = parse_filters("cheapest snacks").merged(f)
    assert merged.sort == "price_asc" and merged.price_max == 10.0 and merged.department == "snacks"


def _rp(
    pid: int,
    name: str,
    price: float | None,
    rating: float | None,
    dept: str = "snacks",
    stock: bool = True,
    count: int = 10,
) -> RetrievedProduct:
    return RetrievedProduct(
        product_id=pid,
        product_name=name,
        aisle="a",
        department=dept,
        price_usd=price,
        avg_rating=rating,
        rating_count=count,
        in_stock=stock,
        score=1.0,
    )


CANDS = [
    _rp(1, "Chips", 9.0, 4.0),
    _rp(2, "Gluten-Free Rotini", 3.0, 4.5, count=50),
    _rp(3, "Cookies", None, None, stock=False),
    _rp(4, "Soda", 2.0, 3.0, dept="beverages"),
]


def test_apply_filter_and_sorts() -> None:
    assert [c.product_id for c in apply_filter(CANDS, StructuredFilter(price_max=5))] == [2, 4]
    assert [c.product_id for c in apply_filter(CANDS, StructuredFilter(price_min=5))] == [1]
    assert [
        c.product_id for c in apply_filter(CANDS, StructuredFilter(department="beverages"))
    ] == [4]
    assert [c.product_id for c in apply_filter(CANDS, StructuredFilter(min_rating=4.2))] == [2]
    assert [c.product_id for c in apply_filter(CANDS, StructuredFilter(in_stock_only=True))] == [
        1,
        2,
        4,
    ]
    assert [c.product_id for c in apply_filter(CANDS, StructuredFilter(diet="gluten_free"))] == [2]
    assert [c.product_id for c in apply_filter(CANDS, StructuredFilter(aisle="zzz"))] == []
    asc = apply_filter(CANDS, StructuredFilter(sort="price_asc"))
    assert [c.product_id for c in asc] == [4, 2, 1, 3] and asc[0].source == "structured"
    assert [c.product_id for c in apply_filter(CANDS, StructuredFilter(sort="price_desc"))][:2] == [
        1,
        2,
    ]
    assert (
        next(c.product_id for c in apply_filter(CANDS, StructuredFilter(sort="rating_desc"))) == 2
    )
    assert (
        next(c.product_id for c in apply_filter(CANDS, StructuredFilter(sort="popularity_desc")))
        == 2
    )


def test_lexically_related_ignores_filter_words() -> None:
    kept = lexically_related("cheapest chips under $5", CANDS)
    assert [c.product_id for c in kept] == [1]
    assert lexically_related("cheapest", CANDS) == list(CANDS)


async def test_in_memory_structured_retriever(catalog: list[dict[str, Any]]) -> None:
    r = InMemoryStructuredRetriever(catalog)
    out = await r.search(StructuredFilter(department="coffee", sort="price_asc"), k=5)
    assert out == []
    out = await r.search(StructuredFilter(aisle="coffee", sort="price_asc"), k=5)
    assert [p.product_id for p in out] == [124, 126, 125]
    assert out[0].source == "structured"


def test_build_sql_parametrised() -> None:
    f = StructuredFilter(
        price_max=5,
        price_min=1,
        department="Snacks",
        aisle="chips",
        min_rating=4,
        in_stock_only=True,
        diet="vegan",
        sort="rating_desc",
    )
    sql, params = build_sql(f, k=7)
    assert sql.count("%s") == len(params) == 9
    assert "price_usd <= %s" in sql and "LOWER(department) = %s" in sql and "in_stock = TRUE" in sql
    assert "product_name ILIKE %s" in sql and sql.endswith("LIMIT %s")
    assert params[0] == 5.0 and params[2] == "snacks" and params[-1] == 7
    sql, params = build_sql(StructuredFilter())
    assert "WHERE" not in sql and params == [20]


async def test_pg_structured_retriever(fake_pool_factory: Any) -> None:
    pool = fake_pool_factory(
        results=[[(1, "A", "a", "d", 1.5, 4.2, 3, True), (2, "B", "a", "d", None, None, 0, False)]]
    )
    r = PgStructuredRetriever(pool)
    out = await r.search(StructuredFilter(price_max=2), k=2)
    assert [p.product_id for p in out] == [1, 2] and out[1].price_usd is None
    sql, params = pool.cursor.executed[0]
    assert "price_usd <= %s" in sql and params == [2.0, 2]
