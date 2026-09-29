# Technical decisions

Each entry states the choice, the alternatives considered and the concrete criterion.

## 1. Qdrant as the primary vector store; pgvector and Chroma as comparison rows

* **Chosen**: Qdrant 1.12 (Docker) for production; pgvector and Chroma implemented behind the
  same `DenseStore` protocol so the benchmark table can compare all three with identical
  embeddings and queries.
* **Why Qdrant**: native HNSW with payload filtering (price/department filters can be pushed
  into the ANN search, which matters for the comparative queries), an async client, and
  horizontal sharding when the catalog outgrows one node. pgvector's IVFFlat/HNSW is adequate
  for 50K rows and keeps carts and vectors in one database, which is the cheaper operational
  choice for a small team; Chroma is embedded, has no server to run and is what the offline
  eval uses today, but it has no multi-node story.
* **Status**: only the Chroma and in-memory rows are measured (no Docker daemon in the dev
  sandbox). The Qdrant/pgvector rows in `eval/RESULTS.md` stay *pendiente* until
  `docker compose up` runs; the code path is tested with mocked clients.

## 2. voyage-3 with MiniLM as the free alternative — and an explicit hash fallback

* **Chosen**: `voyage-3` (1024-d, $0.12/M tokens) when `VOYAGE_API_KEY` exists;
  `sentence-transformers/all-MiniLM-L6-v2` padded to 1024-d behind the optional `ml` extra;
  `HashEmbeddings` (hashed words + trigrams) as the always-available default.
* **Why not ship MiniLM in the base install**: it drags torch (2-6 GB). The base install is
  700 MB, boots in CI and on a laptop, and the extra is one `pip install -e ".[ml]"` away.
* **Why a hash fallback at all**: the API must answer deterministically without any model
  (CI, the static demo, the red-team target). The hash vector is a lexical fingerprint, not a
  semantic embedding; every report produced with it is labelled *deterministic fallback* and
  the semantic category of the eval shows exactly where it fails (`docs/error_analysis.md`).

## 3. Reciprocal Rank Fusion instead of learned or weighted score fusion

BM25 scores and cosine similarities live on different scales; RRF with `k=60` needs no
normalisation and no tuning, and the ablation shows it never loses to either input on nDCG@10
(0.793 vs 0.773 BM25 and 0.790 dense on the fallback run). Weighted fusion would need a
per-embedding calibration that changes every time the embedding backend changes.

## 4. Reranking, then structured filters, gated by lexical relatedness

Filters (`under $5`, `best rated`) are applied **after** semantic retrieval, on the reranked
pool first and the fused list second, and only on candidates that share a descriptive token
with the query. Without the gate, "cheapest coffee" sorted cat litter first because sorting
the whole fused list by price ignores relevance. If nothing survives, the structured
retriever queries the catalog directly (SQL on Postgres or the in-memory filter) and the reply
says so. This is why `+ rerank + filtros` moves the comparative nDCG@10 from 0.754 to 0.948.

## 5. LangGraph with explicit, named branches

The flow has five branches after intent classification and each needs different state
(retrieved products, cart, refund decision). `StateGraph` makes the transitions visible and
testable (`route_after_classify` is a pure function). Node names must not collide with state
keys; this project renamed `cart` → `cart_ops`, `refund` → `refund_policy`, etc.

## 6. Deterministic fallback for every paid dependency

Every external call (Claude, Voyage, Cohere, Postgres, Redis, Qdrant) sits behind a
`Protocol` with an offline implementation selected by env vars. Benefits: tests never spend
credits, `pytest` runs in 20 s, the demo works on a fresh machine, and a rate-limited provider
degrades to a fallback instead of a 500. The trade-off is that fallback numbers must never be
reported as production numbers, which the eval enforces through the `provenance` block.

## 7. Escalation policy for refunds (project-specific DoD item)

`src/agents/refund_handler.py` auto-approves a refund only when **both** hold:

1. amount ≤ `ESCALATION_REFUND_THRESHOLD_USD` (default **$100**), and
2. no department involved is in `SENSITIVE_DEPARTMENTS = {alcohol, tobacco, pharmacy, babies}`.

The amount comes, in order, from an explicit figure in the message ("refund my $200 order"),
the order referenced by number, the product named ("return the ketchup"), or the cart total.
Departments come from the order, the named product and the cart lines. Any explicit request
for a human, a manager or a complaint word routes to `escalate_to_human` regardless of amount.
Rationale: regulated goods (alcohol, tobacco, pharmacy) and infant products carry legal or
safety exposure that an automated refund should not absorb; $100 caps the blast radius of a
wrong automatic decision at roughly one weekly basket.

## 8. Model pinned to a dated id

`ANTHROPIC_MODEL=claude-sonnet-4-5-20250929`. Aliases such as `claude-sonnet-4-5` move under
you; a dated id makes an eval run reproducible and lets the cost table quote one price.

## 9. Session state in Redis with a TTL, carts in Postgres

Conversational context (history, last shown products) is small, per session and disposable:
Redis with `SESSION_TTL_SECONDS` (30 min) fits. Carts have money attached and must survive a
restart: Postgres with a `UNIQUE (session_id, product_id)` constraint and one transaction per
operation. Both have in-memory twins for tests.

## 10. Coverage floor at 70 %, measured at 97 %

The floor guards against silent deletion of helpers; the extra coverage came from testing the
Claude paths with fake clients rather than skipping them. Integration tests hit the real graph
and the real FastAPI app through `TestClient`; only network and databases are faked.

## 11. Ground truth on the synthetic catalog, declared as such

Instacart needs Kaggle credentials that are not available here. Labelling against the
synthetic catalog keeps the eval reproducible today; the queries do not change when the real
catalog arrives, only the `product_id` lists (see `data/eval/README.md`).
