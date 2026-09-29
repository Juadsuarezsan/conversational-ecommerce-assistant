"""FastAPI application: ``/health``, ``/api/chat`` and the cart / product endpoints.

The app boots with no API keys, no Docker and no ML extras (in-memory store, hash
embeddings, lexical reranker, template synthesizer) and then answers
deterministically, which is what the red-team project uses it for.

No ``from __future__ import annotations`` here: FastAPI resolves the route
signatures at import time and the routes are defined inside a factory closure.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src import __version__
from src.agents.orchestrator import build_graph, next_session_state, run_turn
from src.agents.runtime import AgentRuntime, build_runtime
from src.api.schemas import (
    CartAddRequest,
    CartItem,
    CartView,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RetrievedProduct,
)
from src.config import get_settings
from src.observability import configure_logging, configure_tracing
from src.retrieval.hybrid import HybridRetriever

limiter = Limiter(key_func=get_remote_address, default_limits=[get_settings().rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the runtime once per process and close it on shutdown."""
    settings = get_settings()
    configure_logging()
    configure_tracing(settings)
    rt = await build_runtime(settings)
    rt.graph = build_graph(rt)
    app.state.rt = rt
    logger.info(
        "API ready: model={} llm_enabled={} store={} cart={} session={}",
        settings.anthropic_model,
        settings.llm_enabled,
        rt.dense.name,
        settings.cart_backend,
        settings.session_backend,
    )
    try:
        yield
    finally:
        await rt.close()


def create_app() -> FastAPI:
    """Application factory (used by uvicorn and by the tests)."""
    settings = get_settings()
    app = FastAPI(
        title="Conversational E-commerce Assistant",
        version=__version__,
        description=(
            "Hybrid RAG (BM25 + dense + RRF + rerank) over an Instacart-like catalog "
            "with a multi-turn LangGraph cart agent."
        ),
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    _register_routes(app)
    return app


def _rt(request: Request) -> AgentRuntime:
    rt: AgentRuntime = request.app.state.rt
    return rt


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        s = get_settings()
        rt = _rt(request)
        return HealthResponse(
            status="ok",
            version=__version__,
            model=s.anthropic_model,
            llm_enabled=s.llm_enabled,
            embedding_backend=rt.embedder.name,
            rerank_backend=rt.reranker.name,
            vector_store=rt.dense.name,
            cart_backend=s.cart_backend,
            session_backend=s.session_backend,
            catalog_size=len(rt.catalog),
        )

    @app.post("/api/chat", response_model=ChatResponse)
    @limiter.limit(get_settings().rate_limit)
    async def chat(request: Request, req: ChatRequest) -> ChatResponse:
        rt = _rt(request)
        session = await rt.session_cache.get(req.session_id)
        try:
            response, state_out = await run_turn(
                rt.graph,
                session_id=req.session_id,
                user_id=req.user_id,
                message=req.message,
                session=session,
                llm_enabled=rt.settings.llm_enabled,
            )
        except Exception as exc:
            logger.exception("chat turn failed")
            raise HTTPException(
                status_code=500, detail="internal error while processing the turn"
            ) from exc
        await rt.session_cache.set(req.session_id, next_session_state(session, response, state_out))
        return response

    @app.get("/api/products/search", response_model=list[RetrievedProduct])
    @limiter.limit(get_settings().rate_limit)
    async def search_products(
        request: Request,
        q: str = Query(..., min_length=1, max_length=200),
        k: int = Query(default=5, ge=1, le=50),
    ) -> list[RetrievedProduct]:
        rt = _rt(request)
        retriever = HybridRetriever(bm25=rt.bm25, dense=rt.dense, embedder=rt.embedder)
        hits = await retriever.search(q, k=max(k, rt.settings.max_retrieval_k))
        return await rt.reranker.rerank(q, hits, top_n=k)

    @app.get("/api/cart/{session_id}", response_model=CartView)
    async def get_cart(request: Request, session_id: str) -> CartView:
        rt = _rt(request)
        items = await rt.cart_mgr.view(session_id)
        return CartView(
            session_id=session_id, items=items, total_usd=await rt.cart_mgr.total_usd(session_id)
        )

    @app.post("/api/cart/{session_id}/items", response_model=CartView)
    async def add_cart_item(request: Request, session_id: str, body: CartAddRequest) -> CartView:
        rt = _rt(request)
        row = rt.bm25.get(body.product_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"product {body.product_id} not found")
        item = CartItem(
            product_id=body.product_id,
            product_name=str(row["product_name"]),
            quantity=body.quantity,
            unit_price_usd=float(row.get("price_usd") or 0.0),
        )
        items = await rt.cart_mgr.add(session_id, item)
        return CartView(
            session_id=session_id, items=items, total_usd=await rt.cart_mgr.total_usd(session_id)
        )

    @app.delete("/api/cart/{session_id}/items/{product_id}", response_model=CartView)
    async def remove_cart_item(request: Request, session_id: str, product_id: int) -> CartView:
        rt = _rt(request)
        items = await rt.cart_mgr.remove(session_id, product_id)
        return CartView(
            session_id=session_id, items=items, total_usd=await rt.cart_mgr.total_usd(session_id)
        )

    @app.post("/api/cart/{session_id}/clear")
    async def clear_cart(request: Request, session_id: str) -> dict[str, Any]:
        rt = _rt(request)
        await rt.cart_mgr.clear(session_id)
        await rt.session_cache.delete(session_id)
        return {"cleared": True, "session_id": session_id}


app = create_app()
