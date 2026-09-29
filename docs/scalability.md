# Scalability

Numbers marked *measured* come from `eval/runs/2026-09-29-full-eval.json` (deterministic
fallback, 213 products, single process, no LLM). Everything else is an estimate with the
assumption stated.

## Where we are (measured)

| Component | Latency | Notes |
|---|---|---|
| BM25 over 213 products | p95 0.3 ms | in-memory `rank_bm25` |
| Dense search, in-memory exact cosine | p95 17.5 ms | pure-Python 213 × 1024 dot products; the bottleneck of the offline stack |
| Dense search, embedded Chroma | p95 4.0 ms | HNSW, same hash embeddings |
| Hybrid (BM25 + dense + RRF) | p95 19.0 ms | dominated by the dense step |
| Rerank (lexical) + filters | p95 0.4 ms | |
| Full conversation of 2-3 turns | p95 38.6 ms | no LLM |

With the LLM enabled each turn adds two Claude calls (router + synthesizer). At ~1-3 s each
that is the p95 driver; retrieval becomes noise.

## 100× (50K products, ~50 req/min)

* **Catalog**: the real Instacart catalog is 49,688 rows. In-memory exact cosine grows
  linearly (≈ 4 s per query in pure Python) — unusable. Switch `VECTOR_STORE=qdrant` (HNSW,
  ~1-5 ms) or `pgvector` (IVFFlat with `lists=100`, ~5-20 ms). Indexing 50K products with
  `voyage-3` costs ≈ 50K × 15 tokens × $0.12/M ≈ **$0.09** once.
* **BM25**: `rank_bm25` at 50K docs is ~20-40 ms per query in Python; acceptable at this
  tier, replace with Postgres `tsvector` + GIN (already in `init.sql` via `pg_trgm`) or
  Qdrant sparse vectors beyond it.
* **API**: FastAPI + uvicorn workers = CPU cores; the graph is async, so waiting on Claude does
  not block. Rate limit per IP (`RATE_LIMIT`) protects the LLM budget.
* **State**: Redis for sessions (≈ 10 KB per session), Postgres for carts with the pool sized
  `DB_POOL_MAX_SIZE=8` per worker.

## 1000× (500 req/min, several tenants)

* **Anthropic rate limits** become the limit before CPU does; request a tier increase and add
  a queue with priority for cart operations (which need no LLM at all in this design: the
  parser is deterministic).
* **Cache query embeddings** by normalised text in Redis; grocery queries repeat heavily.
* **Read replica** for Postgres; keep vectors in Qdrant and only carts/orders in Postgres.
* **Move synthesis to streaming** so perceived latency drops even when p95 does not.

## 10000× and beyond

* Shard Qdrant by tenant or department; Postgres partitioned by tenant.
* Replace the LLM intent router with a fine-tuned small classifier for the high-volume
  intents (the keyword router already handles the offline case; project 02 of this portfolio
  is that classifier).
* Run the eval as a scheduled job against a canary deployment; the API should never run
  `eval` in-process.

## Cost to operate with 1,000 monthly users (estimate)

Assumptions: 6 turns per user per month, 2 Claude calls per turn, ~900 input + 150 output
tokens per call (system prompt + products), `claude-sonnet-4-5-20250929` at $3/M input and
$15/M output.

| Item | Monthly |
|---|---|
| LLM: 12,000 calls × (900 × $3/M + 150 × $15/M) | ≈ **$59** |
| Voyage query embeddings: 6,000 queries × 15 tokens | ≈ $0.01 |
| Cohere rerank: 6,000 queries × $1/1K | ≈ $6 |
| Infra: one small VM (API + Redis) + managed Postgres + Qdrant node | ≈ $60-120 |
| **Total** | **≈ $125-185 / month**, dominated by infrastructure at this scale |

The LLM line scales linearly with turns; the deterministic cart parser keeps roughly a third
of turns LLM-free if the router is trusted on high-confidence keyword matches.

## What is not designed to scale

The eval harness runs single-process on purpose: metrics must be comparable across runs, so
it uses the same fixture, the same seed and the same order every time.
