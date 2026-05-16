"""Dense vector retrieval. Supports Qdrant, pgvector, and Chroma behind the same interface."""
from __future__ import annotations

from typing import Literal, Protocol, Sequence

from loguru import logger

from src.api.schemas import RetrievedProduct
from src.config import get_settings
from src.retrieval.embeddings import get_embedding_provider


class DenseStore(Protocol):
    name: str
    async def index(self, products: Sequence[dict], batch_size: int = 256) -> int: ...
    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]: ...
    async def count(self) -> int: ...
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# QDRANT
# ---------------------------------------------------------------------------

class QdrantStore:
    name = "qdrant"

    def __init__(self, url: str, collection: str = "products") -> None:
        from qdrant_client import AsyncQdrantClient
        self.client = AsyncQdrantClient(url=url)
        self.collection = collection
        self._dim = 1024

    async def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams
        cols = await self.client.get_collections()
        names = {c.name for c in cols.collections}
        if self.collection not in names:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

    async def index(self, products: Sequence[dict], batch_size: int = 256) -> int:
        from qdrant_client.models import PointStruct
        await self.ensure_collection()
        embedder = get_embedding_provider()
        total = 0
        for i in range(0, len(products), batch_size):
            batch = list(products[i : i + batch_size])
            texts = [f"{p['product_name']} | {p.get('aisle','')} | {p.get('department','')}" for p in batch]
            embeddings = await embedder.embed_documents(texts)
            points = [
                PointStruct(
                    id=int(p["product_id"]),
                    vector=emb,
                    payload={
                        "product_id": int(p["product_id"]),
                        "product_name": p["product_name"],
                        "aisle": p.get("aisle", ""),
                        "department": p.get("department", ""),
                        "price_usd": p.get("price_usd"),
                        "avg_rating": p.get("avg_rating"),
                        "rating_count": p.get("rating_count", 0),
                        "in_stock": p.get("in_stock", True),
                    },
                )
                for p, emb in zip(batch, embeddings)
            ]
            await self.client.upsert(collection_name=self.collection, points=points)
            total += len(points)
        return total

    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]:
        result = await self.client.search(
            collection_name=self.collection, query_vector=query_embedding, limit=k,
        )
        return [
            RetrievedProduct(
                product_id=r.payload["product_id"],
                product_name=r.payload["product_name"],
                aisle=r.payload.get("aisle", ""),
                department=r.payload.get("department", ""),
                price_usd=r.payload.get("price_usd"),
                avg_rating=r.payload.get("avg_rating"),
                rating_count=r.payload.get("rating_count", 0),
                in_stock=r.payload.get("in_stock", True),
                score=float(r.score),
                source="dense",
            )
            for r in result
        ]

    async def count(self) -> int:
        info = await self.client.get_collection(self.collection)
        return int(info.points_count or 0)

    async def close(self) -> None:
        await self.client.close()


# ---------------------------------------------------------------------------
# PGVECTOR  (uses psycopg async + pgvector helper)
# ---------------------------------------------------------------------------

class PgVectorStore:
    name = "pgvector"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn.replace("postgresql+psycopg://", "postgresql://")

    async def _conn(self):
        import psycopg
        from pgvector.psycopg import register_vector_async
        conn = await psycopg.AsyncConnection.connect(self.dsn)
        await register_vector_async(conn)
        return conn

    async def index(self, products: Sequence[dict], batch_size: int = 256) -> int:
        embedder = get_embedding_provider()
        total = 0
        conn = await self._conn()
        try:
            async with conn.cursor() as cur:
                for i in range(0, len(products), batch_size):
                    batch = list(products[i : i + batch_size])
                    texts = [f"{p['product_name']} | {p.get('aisle','')} | {p.get('department','')}" for p in batch]
                    embeddings = await embedder.embed_documents(texts)
                    for p, emb in zip(batch, embeddings):
                        await cur.execute(
                            """
                            UPDATE products
                               SET embedding = %s
                             WHERE product_id = %s
                            """,
                            (emb, int(p["product_id"])),
                        )
                    total += len(batch)
            await conn.commit()
        finally:
            await conn.close()
        return total

    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]:
        conn = await self._conn()
        try:
            async with conn.cursor() as cur:
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
        finally:
            await conn.close()
        return [
            RetrievedProduct(
                product_id=r[0], product_name=r[1], aisle=r[2], department=r[3],
                price_usd=float(r[4]) if r[4] is not None else None,
                avg_rating=float(r[5]) if r[5] is not None else None,
                rating_count=r[6], in_stock=r[7], score=float(r[8]), source="dense",
            )
            for r in rows
        ]

    async def count(self) -> int:
        conn = await self._conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM products WHERE embedding IS NOT NULL")
                row = await cur.fetchone()
                return int(row[0])
        finally:
            await conn.close()

    async def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# CHROMA
# ---------------------------------------------------------------------------

class ChromaStore:
    name = "chroma"

    def __init__(self, persist_dir: str = "./data/chroma") -> None:
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="products", metadata={"hnsw:space": "cosine"})

    async def index(self, products: Sequence[dict], batch_size: int = 256) -> int:
        embedder = get_embedding_provider()
        total = 0
        for i in range(0, len(products), batch_size):
            batch = list(products[i : i + batch_size])
            texts = [f"{p['product_name']} | {p.get('aisle','')} | {p.get('department','')}" for p in batch]
            embeddings = await embedder.embed_documents(texts)
            self.collection.upsert(
                ids=[str(p["product_id"]) for p in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{
                    "product_id": int(p["product_id"]),
                    "product_name": p["product_name"],
                    "aisle": p.get("aisle", ""),
                    "department": p.get("department", ""),
                    "price_usd": p.get("price_usd"),
                    "avg_rating": p.get("avg_rating"),
                    "rating_count": p.get("rating_count", 0),
                    "in_stock": p.get("in_stock", True),
                } for p in batch],
            )
            total += len(batch)
        return total

    async def search(self, query_embedding: list[float], k: int = 10) -> list[RetrievedProduct]:
        res = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[RetrievedProduct] = []
        for _id, meta, dist in zip(ids, metas, dists):
            out.append(RetrievedProduct(
                product_id=int(meta["product_id"]),
                product_name=meta["product_name"],
                aisle=meta.get("aisle", ""),
                department=meta.get("department", ""),
                price_usd=meta.get("price_usd"),
                avg_rating=meta.get("avg_rating"),
                rating_count=meta.get("rating_count", 0),
                in_stock=meta.get("in_stock", True),
                score=1.0 - float(dist),  # cosine distance → similarity
                source="dense",
            ))
        return out

    async def count(self) -> int:
        return self.collection.count()

    async def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# FACTORY
# ---------------------------------------------------------------------------

def build_store(kind: Literal["qdrant", "pgvector", "chroma"]) -> DenseStore:
    s = get_settings()
    if kind == "qdrant":
        return QdrantStore(s.qdrant_url)
    if kind == "pgvector":
        return PgVectorStore(s.database_url)
    if kind == "chroma":
        return ChromaStore()
    raise ValueError(f"Unknown dense store: {kind}")
