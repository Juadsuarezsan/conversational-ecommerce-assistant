# Data schema

## Source dataset

**Instacart Market Basket Analysis** — Kaggle,
https://www.kaggle.com/competitions/instacart-market-basket-analysis/data. Free for
non-commercial and academic use under the competition rules. Files used: `products.csv`
(49,688 rows), `aisles.csv` (134), `departments.csv` (21). Orders (3.4 M rows) are not used by
this project. Download with `python scripts/download_data.py` (Kaggle CLI or `--url` mirror);
the raw folder is git-ignored and every CSV's SHA-256 is appended to `data/MANIFEST.txt`.

When the raw files are absent, `python -m src.ingestion.preprocess` builds the same schema from
the **synthetic catalog** (`src/ingestion/synthetic_catalog.py`, 213 products, 21 departments,
`RNG_SEED = 20260516`). The manifest line says which source produced the processed file.

## `data/processed/products.csv` (and table `products`)

| Column | Type | Range / values | Source | Description |
|---|---|---|---|---|
| `product_id` | int | 1..49688 (real) / 1..213 (synthetic) | Instacart | Primary key; ids 1-12 of the synthetic catalog are hand-written anchor products |
| `product_name` | str | ≤ 160 chars | Instacart | Display name; indexed by BM25 and embedded together with aisle and department |
| `aisle_id` | int | 1..134 | Instacart | Foreign key to `aisles.csv` |
| `aisle` | str | 134 values (e.g. `cereal`, `milk`) | Instacart | Denormalised aisle name |
| `department_id` | int | 1..21 | Instacart | Foreign key to `departments.csv` |
| `department` | str | 21 values (e.g. `dairy eggs`, `alcohol`) | Instacart | Denormalised department; `alcohol`, `babies` trigger the refund escalation policy |
| `price_usd` | float | 1.49..24.99 | **synthetic** | Uniform draw, seed 20260516; Instacart has no prices |
| `avg_rating` | float | 3.80..4.90 | **synthetic** | Uniform draw |
| `rating_count` | int | 50..6000 | **synthetic** | Uniform integer draw; used for the *popularity* sort |
| `in_stock` | bool | 95 % `True` | **synthetic** | Bernoulli(0.95) |
| `embedding` | vector(1024) | L2-normalised | computed | Only in Postgres/Qdrant/Chroma; provider selected by `EMBEDDING_BACKEND` |

The synthetic columns are generated identically for the real and synthetic catalogs so the
eval, the filters and the demo behave the same. They are **not** real market data and every
document that shows them says so.

## `data/eval/queries.jsonl`

100 records; schema and labelling rules in `data/eval/README.md`.

| Field | Type | Description |
|---|---|---|
| `id` | str | `q-<category>-NN` |
| `category` | enum | `lookup` (30), `semantic` (30), `comparative` (20), `multi_turn` (20) |
| `query` | str | The user message (final turn for multi-turn) |
| `relevant` | list | `{product_id, relevance ∈ {1,2,3}}` hand-labelled against the synthetic catalog |
| `setup_turns` | list[str] | multi-turn only: messages sent first in the same session |
| `context` | str | multi-turn only: human-readable description of the situation |
| `expect` | object | multi-turn only: deterministic checks (`intent`, `escalated`, `cart_has`, `cart_has_rank`, `qty_rank`, `cart_qty_rank`, `cart_size`, `cart_empty`, `response_contains`, `response_mentions_any`) |

## Conversational state (`src/db/init.sql`)

| Table | Key columns | Purpose |
|---|---|---|
| `sessions` | `session_id` PK, `user_id`, `state` JSONB, timestamps | One row per conversation |
| `cart_items` | `id`, `session_id` FK, `product_id` FK, `quantity > 0`, `UNIQUE (session_id, product_id)` | Cart lines; `PgCartManager` merges quantities on add |
| `orders` | `order_id` PK, `user_id`, `total_usd`, `status ∈ {pending, paid, shipped, delivered, refunded}`, `items` JSONB | Order history for `order_status` and refunds |
| `audit_log` | `session_id`, `trace_id`, `turn`, `intent`, `confidence`, `query`, `retrieved_ids`, `response`, `escalated`, `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd` | One row per agent turn for replay and analysis |

## Splits

There is no training in this project; the eval set is the only labelled artefact and it is
used whole. Its seed for any sampling is the same `20260516` used everywhere else.

## Reproducibility

```bash
python scripts/download_data.py            # optional, needs Kaggle credentials
python -m src.ingestion.preprocess         # data/processed/products.csv + MANIFEST entry
sha256sum -c <(awk '{print $1"  "$2}' data/MANIFEST.txt)
```
