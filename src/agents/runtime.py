"""Assemble the retrieval indexes, stores and caches the agent needs at run time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.agents.cart_manager import CartManager, InMemoryCartManager, PgCartManager, open_pg_pool
from src.agents.orders import InMemoryOrderStore, OrderStore
from src.config import Settings, get_settings
from src.ingestion.synthetic_catalog import build_catalog
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseStore, build_store
from src.retrieval.embeddings import EmbeddingProvider, get_embedding_provider
from src.retrieval.rerank import Reranker, get_reranker
from src.retrieval.structured import InMemoryStructuredRetriever, PgStructuredRetriever
from src.session.cache import InMemorySessionCache, SessionCache, build_redis_cache


@dataclass
class AgentRuntime:
    """Everything :func:`src.agents.orchestrator.build_graph` needs, wired once per process."""

    settings: Settings
    catalog: list[dict[str, Any]]
    bm25: BM25Index
    dense: DenseStore
    embedder: EmbeddingProvider
    reranker: Reranker
    structured: Any
    cart_mgr: CartManager
    session_cache: SessionCache
    orders: OrderStore
    graph: Any = None
    _pool: Any = field(default=None, repr=False)

    async def close(self) -> None:
        """Release connections held by stores and pools."""
        await self.dense.close()
        if self._pool is not None:
            await self._pool.close()


async def build_runtime(
    settings: Settings | None = None,
    catalog: Sequence[dict[str, Any]] | None = None,
    *,
    embedder: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
) -> AgentRuntime:
    """Build indexes and stores according to ``settings`` (offline defaults need nothing).

    Args:
        settings: Settings; defaults to :func:`get_settings`.
        catalog: Product dicts; defaults to the synthetic catalog.
        embedder: Injected embedding provider (tests); defaults to the configured one.
        reranker: Injected reranker (tests); defaults to the configured one.
    """
    s = settings or get_settings()
    products = list(catalog) if catalog is not None else build_catalog()
    logger.info("Catalog: {} products", len(products))

    emb = embedder or get_embedding_provider()
    rr = reranker or get_reranker()
    bm25 = BM25Index(products)
    dense = build_store(s.vector_store, embedder=emb)
    indexed = await dense.index(products)
    logger.info("Indexed {} products into {} with {} embeddings", indexed, dense.name, emb.name)

    pool: Any = None
    cart_mgr: CartManager
    structured: Any
    if s.cart_backend == "postgres":
        pool = await open_pg_pool(
            s.database_url,
            min_size=s.db_pool_min_size,
            max_size=s.db_pool_max_size,
            timeout=s.external_timeout_seconds,
        )
        cart_mgr = PgCartManager(pool)
        structured = PgStructuredRetriever(pool)
    else:
        cart_mgr = InMemoryCartManager()
        structured = InMemoryStructuredRetriever(products)

    session_cache: SessionCache
    if s.session_backend == "redis":
        session_cache = build_redis_cache(
            s.redis_url, s.session_ttl_seconds, timeout=s.external_timeout_seconds
        )
    else:
        session_cache = InMemorySessionCache(ttl_seconds=s.session_ttl_seconds)

    return AgentRuntime(
        settings=s,
        catalog=products,
        bm25=bm25,
        dense=dense,
        embedder=emb,
        reranker=rr,
        structured=structured,
        cart_mgr=cart_mgr,
        session_cache=session_cache,
        orders=InMemoryOrderStore(),
        _pool=pool,
    )
