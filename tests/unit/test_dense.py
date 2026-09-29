"""Dense stores: in-memory, Qdrant (mocked client), pgvector (fake pool), Chroma (embedded), factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.config import get_settings
from src.retrieval.dense import (
    ChromaStore,
    InMemoryDense,
    PgVectorStore,
    QdrantStore,
    build_store,
    cosine,
    embed_text,
)


def test_cosine() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0], [0.0]) == 0.0


async def test_in_memory_dense_roundtrip(catalog: list[dict[str, Any]], hash_embedder: Any) -> None:
    store = InMemoryDense(embedder=hash_embedder)
    assert await store.index(catalog, batch_size=50) == 213
    assert await store.count() == 213
    q = await hash_embedder.embed_query(embed_text(catalog[0]))
    hits = await store.search(q, k=3)
    assert hits[0].product_id == 1 and hits[0].source == "dense" and hits[0].score > 0.99
    await store.close()


async def test_qdrant_store_with_mocked_client(
    mocker: Any, catalog: list[dict[str, Any]], hash_embedder: Any
) -> None:
    client = mocker.MagicMock()
    client.get_collections = mocker.AsyncMock(return_value=mocker.MagicMock(collections=[]))
    client.create_collection = mocker.AsyncMock()
    client.upsert = mocker.AsyncMock()
    client.get_collection = mocker.AsyncMock(return_value=mocker.MagicMock(points_count=3))
    client.close = mocker.AsyncMock()
    hit = mocker.MagicMock(
        score=0.8,
        payload={"product_id": 1, "product_name": "Heinz", "aisle": "a", "department": "d"},
    )
    client.search = mocker.AsyncMock(return_value=[hit])
    mocker.patch("qdrant_client.AsyncQdrantClient", return_value=client)

    store = QdrantStore("http://q:6333", embedder=hash_embedder)
    assert await store.index(catalog[:3]) == 3
    client.create_collection.assert_awaited_once()
    hits = await store.search([0.0] * 1024, k=1)
    assert hits[0].product_id == 1 and hits[0].score == 0.8
    assert await store.count() == 3
    await store.close()


async def test_pgvector_store_with_fake_pool(
    fake_pool_factory: Any, catalog: list[dict[str, Any]], hash_embedder: Any, mocker: Any
) -> None:
    pool = fake_pool_factory(results=[[(1, "Heinz", "a", "d", 4.29, 4.6, 10, True, 0.77)], [(5,)]])
    store = PgVectorStore("postgresql+psycopg://u:p@h/db", embedder=hash_embedder)
    mocker.patch.object(store, "_get_pool", mocker.AsyncMock(return_value=pool))
    assert store.dsn.startswith("postgresql://")
    assert await store.index(catalog[:2]) == 2
    assert any("INSERT INTO products" in s for s, _ in pool.cursor.executed)
    hits = await store.search([0.0] * 1024, k=1)
    assert hits[0].product_id == 1 and hits[0].price_usd == 4.29 and hits[0].score == 0.77
    assert await store.count() == 5
    await store.close()


async def test_pgvector_pool_lifecycle(mocker: Any) -> None:
    pool = mocker.MagicMock()
    pool.open = mocker.AsyncMock()
    pool.close = mocker.AsyncMock()
    mocker.patch("psycopg_pool.AsyncConnectionPool", return_value=pool)
    store = PgVectorStore("postgresql://u:p@h/db")
    assert await store._get_pool() is pool
    assert await store._get_pool() is pool  # cached
    pool.open.assert_awaited_once()
    await store.close()
    pool.close.assert_awaited_once()


@pytest.mark.slow
async def test_chroma_store_embedded(
    tmp_path: Path, catalog: list[dict[str, Any]], hash_embedder: Any
) -> None:
    store = ChromaStore(
        persist_dir=str(tmp_path / "chroma"), collection="test_products", embedder=hash_embedder
    )
    assert await store.index(catalog[:20], batch_size=8) == 20
    assert await store.count() == 20
    q = await hash_embedder.embed_query(embed_text(catalog[0]))
    hits = await store.search(q, k=2)
    assert hits[0].product_id == 1 and hits[0].source == "dense"
    await store.close()


def test_build_store_factory(monkeypatch: pytest.MonkeyPatch, mocker: Any, tmp_path: Path) -> None:
    assert build_store("in_memory").name == "in_memory"
    mocker.patch("qdrant_client.AsyncQdrantClient")
    assert build_store("qdrant").name == "qdrant"
    assert build_store("pgvector").name == "pgvector"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "c"))
    get_settings.cache_clear()
    assert build_store("chroma").name == "chroma"
    with pytest.raises(ValueError):
        build_store("nope")  # type: ignore[arg-type]
