"""FastAPI app — /chat endpoint that drives the LangGraph agent end-to-end."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv()

from src.agents.cart_manager import CartManager
from src.agents.orchestrator import build_graph, run_turn
from src.api.schemas import ChatRequest, ChatResponse
from src.config import get_settings
from src.ingestion.synthetic_catalog import build_catalog
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import build_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading product catalog...")
    catalog = build_catalog()
    logger.info(f"Catalog: {len(catalog)} products")

    bm25 = BM25Index(catalog)

    logger.info(f"Building dense store: {settings.vector_store}")
    dense = build_store(settings.vector_store)
    try:
        indexed = await dense.index(catalog)
        logger.info(f"Indexed {indexed} into {dense.name}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not index into {dense.name}: {exc}. Falling back to in-memory.")
        from src.eval.runner import InMemoryDense
        dense = InMemoryDense()
        await dense.index(catalog)

    cart_mgr = CartManager()
    graph = build_graph(bm25=bm25, dense=dense, cart_mgr=cart_mgr)

    app.state.bm25 = bm25
    app.state.dense = dense
    app.state.cart_mgr = cart_mgr
    app.state.graph = graph
    app.state.history = {}  # session_id -> list[{role, content}]
    yield
    await dense.close()


app = FastAPI(
    title="Conversational E-commerce Assistant",
    version="1.0.0",
    description="Hybrid RAG (BM25+dense+RRF+rerank) over Instacart-like catalog with multi-turn cart agent.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {
        "status": "ok",
        "embedding_backend": s.embedding_backend,
        "rerank_backend": s.rerank_backend,
        "vector_store": s.vector_store,
        "llm_enabled": "yes" if s.anthropic_api_key else "no (heuristic fallback)",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    history = app.state.history.get(req.session_id, [])
    try:
        response = await run_turn(
            app.state.graph,
            session_id=req.session_id, user_id=req.user_id,
            message=req.message, history=history,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    new_history = history + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": response.response},
    ]
    app.state.history[req.session_id] = new_history[-20:]  # keep last 10 turns
    return response


@app.get("/api/cart/{session_id}")
async def get_cart(session_id: str) -> dict:
    return {"items": [i.model_dump() for i in app.state.cart_mgr.view(session_id)],
            "total_usd": app.state.cart_mgr.total_usd(session_id)}


@app.post("/api/cart/{session_id}/clear")
async def clear_cart(session_id: str) -> dict:
    app.state.cart_mgr.clear(session_id)
    return {"cleared": True}


@app.get("/api/eval/run")
async def run_eval_endpoint() -> dict:
    from src.eval.runner import evaluate
    return await evaluate(vector_store="in_memory")
