# Performance profile

Source: `eval/runs/2026-09-29-full-eval.json` (`python -m eval.run --stores in_memory,chroma`,
deterministic fallback, no LLM, Python 3.11, single process, sandbox VM). Latencies are
per-query wall clock measured inside the runner with `time.perf_counter()`.

## Retrieval, 80 single-turn queries

| Stage | p50 ms | p95 ms | Where the time goes |
|---|---|---|---|
| BM25 | 0.2 | 0.3 | `rank_bm25.get_scores` over 213 docs |
| Dense (in-memory) | 17.0 | 17.5 | `HashEmbeddings._vector` (~1 ms) + 213 × 1024 pure-Python cosine (~16 ms) |
| Dense (Chroma) | — | 4.0 | HNSW query; the embedding cost is the same |
| Hybrid | 17.3 | 19.0 | BM25 + dense + RRF; RRF itself is < 0.1 ms |
| Rerank + filters | 0.2 | 0.4 | token overlap over 20 candidates |
| Index build (213 products) | 48 ms | — | `InMemoryDense.index`, hash embeddings |

**Bottleneck**: the exact cosine search in `InMemoryDense.search` is a Python list comprehension
over 1024-d vectors. It is fine for the synthetic catalog and useless for 50K products
(linear in catalog size, ≈ 4 s per query). It stays because it needs no dependency; the
production path is Qdrant or pgvector, and even the embedded Chroma row shows a 4× drop.

Cheap improvements if the in-memory store must grow: store vectors in a NumPy matrix and use one
`matmul` (≈ 0.3 ms for 50K × 1024 float32), or keep hash vectors sparse (they are ~95 % zeros).

## Conversation, 20 scripted multi-turn dialogues (2.7 turns on average)

| Metric | Value |
|---|---|
| p50 / p95 per conversation | 30.3 / 38.6 ms |
| Per turn | ≈ 3 ms for cart / refund / order turns, ≈ 20 ms for search turns |
| Cost | $0 (no LLM) |

With `ANTHROPIC_API_KEY` set, each turn performs two model calls (router, synthesizer) with
`timeout=15 s` and three retries with exponential backoff (1-8 s). Expected p95 is then in the
2-6 s range; it has not been measured here because no key is available, and the cell is marked
*pendiente* in `eval/RESULTS.md`. Token counts and `cost_usd` are already accounted per request
from `usage_metadata` (see `src/observability.py`), so the measurement needs no code change.

## Test suite

126 tests in ≈ 17 s warm (≈ 150 s the first run with coverage, dominated by importing
`chromadb`/`langgraph`). The slowest tests are the eval entry point (3-6 s each) because they
index the catalog and run all 100 queries.

## Batch operations

* Embeddings are computed in batches of 256 (`DenseStore.index(batch_size=256)`).
* `PgVectorStore.index` upserts inside one connection from the pool per batch and commits once.
* The eval loops are sequential by design (comparable latencies); indexing three stores in
  parallel would be the first optimisation for a larger catalog.
