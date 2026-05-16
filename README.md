# Project 01 — Conversational E-commerce Assistant

> Hybrid retrieval (BM25 + dense + RRF) + reranking + multi-turn LangGraph agent over a 50K-product catalog. Customer asks in natural language, manages cart, requests refund. The system decides when to escalate to human.

[![Status](https://img.shields.io/badge/status-planned-fbbf24)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![LLM](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.5-7c5cff)]()
[![Eval](https://img.shields.io/badge/eval-RAGAS-22d3ee)]()

**Industrial use case:** Rappi, Mercado Libre, Walmart, Instacart, Amazon — conversational shopping.

---

## What this project does

A real conversational layer over a real product catalog. The customer types in natural English (or Spanish), the system understands what they want, finds it across 50K products, manages their cart, handles refund logic, and escalates to a human when the model isn't confident enough. Every retrieval and every routing decision is measured against a 100-case hand-labeled eval set.

## Architecture

```
Customer query
   │
   ▼
[Intent Router] Claude with structured output (Pydantic)
   │ classifies: product_search | cart_op | order_status | refund | escalate
   │
   ├──► [Hybrid Retriever]
   │      ├─ BM25 on product_name + aisle + department
   │      ├─ Dense vector search in Qdrant (voyage-3 embeddings)
   │      ├─ Reciprocal Rank Fusion → top 20
   │      └─ Cohere Rerank v3 → top 5
   │
   ├──► [Structured Retriever] SQL on Postgres
   │      └─ filters: price, category, dietary, availability
   │
   ├──► [Cart Manager] Pydantic stateful models
   │      └─ add, remove, update_quantity, view, clear
   │
   ├──► [Refund Handler]
   │      └─ policy: if value > $100 OR sensitive category → escalate
   │
   └──► [Synthesizer] Claude
          └─ response with product_id citations
```

## Status

**Planned.** This README is the brief for Claude Code. The implementation plan follows the spec at `/docs/portafolio_specs.md` (Proyecto 1).

## Roadmap to v1.0.0

1. [ ] **Data pipeline** — `scripts/download_instacart.py` pulls Kaggle dataset, builds processed parquet
2. [ ] **Vector ingestion** — embed product_name + aisle + department with voyage-3, index into Qdrant + pgvector + Chroma in parallel
3. [ ] **Hybrid retrieval module** — BM25 (rank-bm25) + dense + RRF fusion
4. [ ] **Reranking** — Cohere Rerank v3 wrapper with local fallback (`ms-marco-MiniLM-L-6-v2`)
5. [ ] **LangGraph agent** — Intent Router → Retriever → Cart/Refund handlers → Synthesizer
6. [ ] **Eval set** — 100 hand-labeled queries (30 lookup, 30 semantic, 20 comparative, 20 multi-turn)
7. [ ] **RAGAS evaluation** — Faithfulness, Answer Relevance, Context Precision, Context Recall
8. [ ] **Vector DB benchmark** — table comparing Qdrant vs pgvector vs Chroma on Precision@5, Recall@10, nDCG@10, p95 latency, $/1K queries
9. [ ] **Streamlit demo** — multi-turn chat with 5 pre-baked queries, deployed to Streamlit Community Cloud
10. [ ] **Observability** — LangSmith traces public, ≥30 example links in README
11. [ ] **Definition of Done** — all 12 universal blocks + this project's 7 specific items pass

## Stack (locked)

| Layer | Technology |
|---|---|
| Vector store (primary) | Qdrant 1.11+ via Docker |
| Vector stores (compared) | pgvector 0.7+, Chroma 0.5+ |
| Embeddings | Voyage AI `voyage-3` ($0.12 / M tokens) or `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Reranking | Cohere Rerank v3 (`rerank-english-v3.0`, $1 / 1K queries) or `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Hybrid search | rank-bm25 (sparse) + dense + RRF |
| LLM | Claude Sonnet 4.5 (`claude-sonnet-4-5`) via Anthropic API |
| Orchestration | LangGraph 0.2+ StateGraph |
| State | PostgreSQL 16 (carts, orders, audit) |
| Session cache | Redis 7 with configurable TTL |
| Frontend | Streamlit Community Cloud |
| Eval framework | RAGAS 0.2+ |

## Quickstart (target — not yet runnable)

```bash
cd 01-ecommerce-assistant/

# 1. Download dataset (~200 MB, requires Kaggle CLI)
python scripts/download_instacart.py

# 2. Bring up infra
docker compose up -d         # postgres + qdrant + redis

# 3. Ingest products into all 3 vector DBs
python -m src.ingestion.index_products --all-stores

# 4. Run evaluations
python -m src.eval.benchmark_vectordbs    # table per store
python -m src.eval.ragas_eval             # RAG quality metrics

# 5. Start the demo
streamlit run streamlit_app.py
```

## Definition of Done — project-specific

- [ ] Eval set of 100 hand-labeled queries with `product_id` ground truth and relevance scores 1-3
- [ ] Benchmark of 3 vector DBs reported with full metrics table in README
- [ ] Ablation: BM25-only vs dense-only vs hybrid (with and without reranking) reported
- [ ] Streamlit demo deployed publicly with pre-baked queries
- [ ] Multi-turn conversation of ≥6 turns demonstrated in a GIF embedded in README
- [ ] Escalation policy documented in `docs/decisions.md`
- [ ] LangSmith public trace gallery linked from README

Plus the **12 universal blocks** of the Definition of Done in `/docs/portafolio_specs.md`.

## License

MIT.
