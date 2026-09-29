"""Dense vector retrieval: Qdrant, pgvector, Chroma and an in-memory store, one interface.

Every store receives already-computed embeddings through the injected
:class:`EmbeddingProvider`, so the store code is independent from the embedding
backend (hash / local / Voyage).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from loguru import logger

from src.api.schemas import RetrievedProduct
from src.config import VectorStoreKind, get_settings
from src.retrieval.bm25 import to_retrieved
from src.retrieval.embeddings import EMBEDDING_DIM, EmbeddingProvider, get_embedding_provider


class DenseStore(Protocol):
    """A vector store holding one point per product."""

    name: str

    async def index(self, products: Sequence[dict[str, Any]], batch_size: int = 256) -> int:
        """Embed and upsert ``products``; return the number indexed."""
        ...

    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]:
        """Return the ``k`` nearest products to ``query_embedding``."""
        ...

    async def count(self) -> int:
        """Number of indexed points."""
        ...

    async def close(self) -> None:
        """Release connections."""
        ...


def embed_text(p: dict[str, Any]) -> str:
    """Text embedded for a product (same for every store)."""
    return f"{p['product_name']} | {p.get('aisle', '')} | {p.get('department', '')}"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either vector is null."""
    num = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def _payload(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": int(p["product_id"]),
        "product_name": p["product_name"],
        "aisle": p.get("aisle", ""),
        "department": p.get("department", ""),
        "price_usd": p.get("price_usd"),
        "avg_rating": p.get("avg_rating"),
        "rating_count": p.get("rating_count", 0),
        "in_stock": p.get("in_stock", True),
    }


class _EmbedderMixin:
    """Shared lazy access to the embedding provider."""

    _embedder: EmbeddingProvider | None = None

    @property
    def embedder(self) -> EmbeddingProvider:
        """Embedding provider (injected or resolved from settings)."""
        if self._embedder is None:
            self._embedder = get_embedding_provider()
        return self._embedder


# ---------------------------------------------------------------------------
# IN-MEMORY (offline default; exact cosine search)
# ---------------------------------------------------------------------------


class InMemoryDense(_EmbedderMixin):
    """Brute-force cosine search kept in a Python list. Default when no Docker is around."""

    name = "in_memory"

    def __init__(self, embedder: EmbeddingProvider | None = None) -> None:
        self._embedder = embedder
        self._items: list[tuple[int, list[float], dict[str, Any]]] = []

    async def index(self, products: Sequence[dict[str, Any]], batch_size: int = 256) -> int:
        """Embed products in batches and keep them in memory."""
        self._items = []
        for i in range(0, len(products), batch_size):
            batch = list(products[i : i + batch_size])
            embs = await self.embedder.embed_documents([embed_text(p) for p in batch])
            for p, e in zip(batch, embs, strict=True):
                self._items.append((int(p["product_id"]), e, p))
        return len(self._items)

    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]:
        """Exact top-k by cosine similarity (ties broken by product id)."""
        scored = [(cosine(query_embedding, e), -pid, p) for pid, e, p in self._items]
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [to_retrieved(p, s, "dense") for s, _, p in scored[:k]]

    async def count(self) -> int:
        """Number of indexed products."""
        return len(self._items)

    async def close(self) -> None:
        """No-op."""
        return None


# ---------------------------------------------------------------------------
# QDRANT
# ---------------------------------------------------------------------------


class QdrantStore(_EmbedderMixin):
    """Qdrant collection with cosine distance (client imported lazily)."""

    name = "qdrant"

    def __init__(
        self,
        url: str,
        collection: str = "products",
        embedder: EmbeddingProvider | None = None,
        timeout: float = 10.0,
    ) -> None:
        from qdrant_client import AsyncQdrantClient

        self.client = AsyncQdrantClient(url=url, timeout=int(timeout))
        self.collection = collection
        self._embedder = embedder

    async def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        from qdrant_client.models import Distance, VectorParams

        cols = await self.client.get_collections()
        if self.collection not in {c.name for c in cols.collections}:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection {}", self.collection)

    async def index(self, products: Sequence[dict[str, Any]], batch_size: int = 256) -> int:
        """Embed and upsert products in batches."""
        from qdrant_client.models import PointStruct

        await self.ensure_collection()
        total = 0
        for i in range(0, len(products), batch_size):
            batch = list(products[i : i + batch_size])
            embeddings = await self.embedder.embed_documents([embed_text(p) for p in batch])
            points = [
                PointStruct(id=int(p["product_id"]), vector=emb, payload=_payload(p))
                for p, emb in zip(batch, embeddings, strict=True)
            ]
            await self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        return total

    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]:
        """Nearest neighbours by cosine similarity."""
        result = await self.client.search(
            collection_name=self.collection, query_vector=query_embedding, limit=k
        )
        return [to_retrieved(dict(r.payload or {}), float(r.score), "dense") for r in result]

    async def count(self) -> int:
        """Number of points in the collection."""
        info = await self.client.get_collection(self.collection)
        return int(info.points_count or 0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.close()


# ---------------------------------------------------------------------------
# PGVECTOR (psycopg3 async pool)
# ---------------------------------------------------------------------------


class PgVectorStore(_EmbedderMixin):
    """pgvector over the ``products`` table using a psycopg async connection pool."""

    name = "pgvector"

    def __init__(
        self,
        dsn: str,
        embedder: EmbeddingProvider | None = None,
        min_size: int = 1,
        max_size: int = 8,
        timeout: float = 10.0,
    ) -> None:
        self.dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
        self._embedder = embedder
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            from pgvector.psycopg import register_vector_async
            from psycopg_pool import AsyncConnectionPool

            async def _configure(conn: Any) -> None:
                await register_vector_async(conn)

            self._pool = AsyncConnectionPool(
                self.dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                timeout=self._timeout,
                configure=_configure,
                open=False,
            )
            await self._pool.open()
        return self._pool

    async def index(self, products: Sequence[dict[str, Any]], batch_size: int = 256) -> int:
        """Upsert product rows with their embeddings."""
        pool = await self._get_pool()
        total = 0
        async with pool.connection() as conn, conn.cursor() as cur:
            for i in range(0, len(products), batch_size):
                batch = list(products[i : i + batch_size])
                embeddings = await self.embedder.embed_documents([embed_text(p) for p in batch])
                for p, emb in zip(batch, embeddings, strict=True):
                    await cur.execute(
                        """
                        INSERT INTO products (product_id, product_name, aisle_id, aisle,
                                              department_id, department, price_usd, avg_rating,
                                              rating_count, in_stock, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (product_id) DO UPDATE
                           SET embedding = EXCLUDED.embedding,
                               price_usd = EXCLUDED.price_usd,
                               avg_rating = EXCLUDED.avg_rating,
                               rating_count = EXCLUDED.rating_count,
                               in_stock = EXCLUDED.in_stock
                        """,
                        (
                            int(p["product_id"]),
                            p["product_name"],
                            int(p.get("aisle_id", 0) or 0),
                            p.get("aisle", ""),
                            int(p.get("department_id", 0) or 0),
                            p.get("department", ""),
                            p.get("price_usd"),
                            p.get("avg_rating"),
                            int(p.get("rating_count", 0) or 0),
                            bool(p.get("in_stock", True)),
                            emb,
                        ),
                    )
                total += len(batch)
            await conn.commit()
        return total

    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]:
        """Cosine nearest neighbours via ``<=>``."""
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT product_id, product_name, aisle, department,
                       price_usd, avg_rating, rating_count, in_stock,
                       1 - (embedding <=> %s::vector) AS similarity
                  FROM products
                 WHERE embedding IS NOT NULL
                 ORDER BY embedding <=> %s::vector
                 LIMIT %s
                """,
                (query_embedding, query_embedding, k),
            )
            rows = await cur.fetchall()
        return [
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
                float(r[8]),
                "dense",
            )
            for r in rows
        ]

    async def count(self) -> int:
        """Rows with a non-null embedding."""
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM products WHERE embedding IS NOT NULL")
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def close(self) -> None:
        """Close the pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


# ---------------------------------------------------------------------------
# CHROMA
# ---------------------------------------------------------------------------


class ChromaStore(_EmbedderMixin):
    """Chroma persistent collection with cosine space (client imported lazily)."""

    name = "chroma"

    def __init__(
        self,
        persist_dir: str = "./data/cache/chroma",
        collection: str = "products",
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = embedder

    async def index(self, products: Sequence[dict[str, Any]], batch_size: int = 256) -> int:
        """Embed and upsert products in batches."""
        total = 0
        for i in range(0, len(products), batch_size):
            batch = list(products[i : i + batch_size])
            texts = [embed_text(p) for p in batch]
            embeddings = await self.embedder.embed_documents(texts)
            self.collection.upsert(
                ids=[str(p["product_id"]) for p in batch],
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=texts,
                metadatas=[{k: v for k, v in _payload(p).items() if v is not None} for p in batch],
            )
            total += len(batch)
        return total

    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]:
        """Nearest neighbours; Chroma returns cosine *distance*, converted to similarity."""
        res = self.collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=k,
        )
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            to_retrieved(dict(meta), 1.0 - float(dist), "dense")
            for meta, dist in zip(metas, dists, strict=True)
        ]

    async def count(self) -> int:
        """Number of points in the collection."""
        return int(self.collection.count())

    async def close(self) -> None:
        """No-op (Chroma persists on write)."""
        return None


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------


def build_store(
    kind: VectorStoreKind, embedder: EmbeddingProvider | None = None, collection: str = "products"
) -> DenseStore:
    """Instantiate the dense store selected by ``VECTOR_STORE``."""
    s = get_settings()
    if kind == "qdrant":
        return QdrantStore(
            s.qdrant_url,
            collection=collection,
            embedder=embedder,
            timeout=s.external_timeout_seconds,
        )
    if kind == "pgvector":
        return PgVectorStore(
            s.database_url,
            embedder=embedder,
            min_size=s.db_pool_min_size,
            max_size=s.db_pool_max_size,
            timeout=s.external_timeout_seconds,
        )
    if kind == "chroma":
        return ChromaStore(s.chroma_persist_dir, collection=collection, embedder=embedder)
    if kind == "in_memory":
        return InMemoryDense(embedder=embedder)
    raise ValueError(f"Unknown dense store: {kind}")
