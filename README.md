# Conversational E-commerce Assistant

[![CI](https://github.com/Juadsuarezsan/conversational-ecommerce-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Juadsuarezsan/conversational-ecommerce-assistant/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](.python-version)
[![LLM](https://img.shields.io/badge/LLM-claude--sonnet--4--5--20250929-7c5cff)](docs/decisions.md)

**Demo:** static page in [`demo/index.html`](demo/index.html) (replays cases produced by this code; switch to *live* to call the API). Streamlit Community Cloud deployment: pending (see [Status](#status)).

## What this project does

A shopping assistant you can talk to. You ask for products in plain English ("healthy breakfast
for kids", "top rated cereals under $5"), it finds them in the catalog, adds them to your cart
when you say "add 2 of the second one", answers "where is my order #1001", decides whether a
refund can be approved automatically, and hands the conversation to a human when the request
is too large, too sensitive or explicitly asks for one. Every product it mentions is cited by
id so the answer can be checked against the catalog.

## Industrial use case

Any B2C company with a large catalog and a chat channel: online grocers (Instacart, Rappi),
marketplaces (Mercado Libre, Amazon) and retailers with e-commerce (Walmart). The recurring
requirements are the same: search that understands intent rather than keywords, a cart that
follows the conversation ("that one", "make it three"), refund rules that a compliance team can
read, and a measurable handoff to humans. This repository is a reference implementation of
those four pieces with an evaluation harness attached.

## Architecture

![Architecture](docs/architecture.svg)

* **Intent Router** — Claude with a strict JSON schema (or a keyword router without a key)
  classifies `product_search | cart_op | order_status | refund | escalate | greeting` and
  extracts filters.
* **Hybrid Retriever** — BM25 over name + aisle + department, dense vectors in Qdrant /
  pgvector / Chroma / in-memory, Reciprocal Rank Fusion (k = 60), reranking (Cohere v3,
  cross-encoder, or a lexical fallback), then price / rating / stock / diet filters.
* **Structured Retriever** — the same filters compiled to parametrised SQL on Postgres or
  applied in memory, used when no semantic candidate satisfies the constraint.
* **Cart Manager** — `CartManager` protocol with in-memory and Postgres implementations; cart
  operations are parsed from text and references are resolved against the previous results.
* **Refund Handler** — auto-approve only if amount ≤ $100 **and** no sensitive department
  (alcohol, tobacco, pharmacy, babies); otherwise escalate. Policy in
  [`docs/decisions.md`](docs/decisions.md#7-escalation-policy-for-refunds-project-specific-dod-item).
* **Synthesizer** — Claude (or a template) writes the reply with `[pid:N]` citations.
* **Orchestration** — LangGraph `StateGraph`; every node logs its input/output with the request
  `trace_id`; tokens and `cost_usd` are accounted per request; LangSmith is enabled by env var.

Details: [`docs/architecture.md`](docs/architecture.md).

## Metrics and results

Everything below is copied from [`eval/RESULTS.md`](eval/RESULTS.md), which
`python -m eval.run` regenerates from a run stored in [`eval/runs/`](eval/runs/). The run was made
in **deterministic fallback mode** (hash embeddings, lexical reranker, no LLM) on the
**synthetic catalog** (213 products); those rows measure the fallback, not the production stack.

**Vector DB table (spec):**

| Vector DB | Precision@5 | Recall@10 | nDCG@10 | Latency p95 (dense, ms) | Cost / 1K queries |
|---|---|---|---|---|---|
| Qdrant | pending (needs Docker) | pending | pending | pending | $0 self-hosted |
| pgvector | pending (needs Docker) | pending | pending | pending | $0 self-hosted |
| Chroma (embedded) | 0.398 | 0.874 | 0.792 | 4.0 | $0 embedded |
| in-memory (fallback) | 0.398 | 0.874 | 0.790 | 17.5 | $0 |

**Ablation (80 single-turn queries, in-memory store, fallback components):**

| Strategy | P@5 | R@10 | MRR | nDCG@10 | p95 ms |
|---|---|---|---|---|---|
| BM25 only (no-AI baseline) | 0.383 | 0.861 | 0.825 | 0.773 | 0.3 |
| Dense only | 0.398 | 0.874 | 0.848 | 0.790 | 17.5 |
| Hybrid BM25 + dense (RRF) | 0.405 | 0.877 | 0.848 | 0.793 | 19.0 |
| Hybrid + rerank + filters | 0.393 | 0.867 | **0.866** | **0.829** | 0.4 |
| Zero-shot Claude, no retrieval | pending (requires `ANTHROPIC_API_KEY`) | | | | |

nDCG@10 by category with the full stack: lookup **0.992**, comparative **0.948**, semantic
**0.586** — the semantic gap is the fallback's known weakness, analysed case by case in
[`docs/error_analysis.md`](docs/error_analysis.md).

**Conversational (20 scripted multi-turn dialogues, deterministic checks on cart and intent):**
task completion **20/20**, 2.7 turns per task on average, p95 38.6 ms per conversation, $0.
LLM-as-judge (rubric in `src/eval/judge.py`) and RAGAS (Faithfulness, Answer Relevance,
Context Precision, Context Recall) are wired and **pending an `ANTHROPIC_API_KEY`**.

## Quickstart

```bash
git clone https://github.com/Juadsuarezsan/conversational-ecommerce-assistant.git
cd conversational-ecommerce-assistant
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"    # ~700 MB, no torch
.venv/bin/uvicorn src.api.main:app --port 8000                 # boots with no keys, no Docker
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"session_id":"s1","message":"lactose free milk gallon"}'
.venv/bin/python -m eval.run                                   # regenerates eval/RESULTS.md
```

Optional: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (LLM router + synthesizer),
`VOYAGE_API_KEY` / `COHERE_API_KEY` (production embeddings and reranking), or
`pip install -e ".[ml]"` with `EMBEDDING_BACKEND=local RERANK_BACKEND=local` for the free
MiniLM + cross-encoder path. `docker compose up` starts Postgres + pgvector, Qdrant, Redis, the
API and the Streamlit UI (`streamlit_app.py`); validation of the compose stack is pending
(no Docker daemon in the development sandbox).

`make test lint typecheck eval` runs the same gates as CI.

## Technical decisions

1. **Qdrant as primary store, pgvector and Chroma behind the same protocol** so the benchmark
   compares them with identical embeddings; Qdrant for filtered ANN and sharding, pgvector for
   one-database simplicity, Chroma for zero-ops local runs.
2. **voyage-3 in production, MiniLM as the free extra, hash embeddings as the always-on
   fallback** — the base install stays at 700 MB and boots in CI; fallback numbers are always
   labelled.
3. **RRF instead of weighted fusion** — no score normalisation, no per-embedding tuning, never
   worse than either input on nDCG@10 in the ablation.
4. **Filters after reranking, gated by lexical relatedness** — moved comparative nDCG@10 from
   0.754 to 0.948 and stopped "cheapest coffee" from returning cat litter.
5. **LangGraph with node names distinct from state keys** — explicit, unit-testable routing and
   the fix for the node/state-key collision that broke three sibling projects.
6. **Deterministic fallback behind every paid dependency** — tests never spend credits, the demo
   runs on a fresh machine, and provenance is recorded in every eval run.
7. **Model pinned to `claude-sonnet-4-5-20250929`**, timeouts and tenacity on every external call.

Full list with alternatives in [`docs/decisions.md`](docs/decisions.md).

## Known limitations

* **Ground truth is on a synthetic catalog** (213 products); Instacart needs Kaggle credentials.
  Prices, ratings and stock are random draws, not market data.
* **Fallback retrieval is lexical**: semantic queries ("things to make guacamole") fail without
  a neural embedding; nDCG@10 0.586 on that category.
* **No LLM numbers yet**: RAGAS, LLM-judge, zero-shot baseline and LLM latency/cost are pending
  an API key; the deterministic conversation checks are not a substitute for a judge.
* **Qdrant / pgvector rows are pending Docker**; the code paths are covered by mocked tests only.
* **English only** in the keyword router and cart parser; the LLM path handles Spanish.
* **Orders are a demo store**; there is no checkout flow, so `order_status` answers three
  fixed orders.
* **Swap semantics** ("swap that for X") add the new product but do not remove the old one.

## Future work (priority order)

1. Run the eval with `local` and then `voyage`/`cohere` backends and Docker stores to fill the
   pending rows; publish LangSmith traces.
2. Enrich the catalog with diet / nutrition attributes (Open Food Facts) so `dairy_free`,
   `low_carb` filters become deterministic.
3. Deploy the Streamlit UI to Community Cloud against a hosted API; record the 6-turn GIF.
4. Add a checkout flow and persist orders + audit log through Postgres.
5. Query rewriting for semantic queries (LLM expands "guacamole" to ingredients) with a cache.

## Repository map

```
src/agents/      intent router, cart manager + ops, refund policy, orders, synthesizer, LangGraph
src/retrieval/   bm25, embeddings, dense stores, hybrid RRF, rerankers, structured filters
src/session/     session cache (in-memory, Redis)
src/api/         FastAPI app + schemas          src/eval/   metrics, runner, conversation, judge, ragas, baselines
src/ingestion/   synthetic catalog, preprocess  eval/run.py single-command evaluation → eval/RESULTS.md
data/eval/       100 hand-labelled queries      docs/       architecture, decisions, scalability, performance, error analysis, data schema, blog
demo/            static demo + baked cases      scripts/    download_data.py, bake_demo.py
tests/           126 tests, 97 % coverage       notebooks/  demo.ipynb
```

## Security and quality gates

`ruff check .`, `black --check .`, `mypy --strict src/`, `pytest --cov=src --cov-fail-under=70`
(97.6 % measured) and `gitleaks detect --no-banner --redact` (0 findings, 2026-09-29) run in CI
on every push and pull request. CORS origins and the rate limit are environment variables;
inputs are validated with Pydantic (422 on empty, oversized or malformed bodies).

## Status

Done: hybrid retrieval with four store backends, LangGraph agent with real cart / refund / order
handling, session cache, 100-query eval set, reproducible eval with provenance labelling, CI,
static demo, Streamlit app, docs. Pending keys or infrastructure: Instacart download, Docker
benchmark rows, RAGAS and LLM-judge, LangSmith traces, Streamlit Cloud deployment and GIF.

## Dataset and license

Instacart Market Basket Analysis, Kaggle
(https://www.kaggle.com/competitions/instacart-market-basket-analysis/data), free for
non-commercial and academic use. Schema in [`docs/data_schema.md`](docs/data_schema.md).
Code under the [MIT License](LICENSE).

## Author

**Juan David Suárez Sánchez** · juadsuarezsan@unal.edu.co ·
[LinkedIn](https://www.linkedin.com/in/juan-david-suarez-sanchez-31ab281b7)
