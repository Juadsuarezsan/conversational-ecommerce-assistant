# Eval set

Hand-labeled queries with product_id relevance scores.

**Current seed: 30 cases.** The DoD target is 100. Scaling to 100 requires either:
- Pulling the real Instacart dataset (Kaggle credentials) and labeling against it, or
- Generating templated variations with Claude and validating each manually.

## Categories

| Category | Count | Description |
|---|---|---|
| `lookup` | 8 | Direct product name / brand queries |
| `semantic` | 7 | Intent-based queries that need semantic understanding |
| `comparative` | 4 | "best", "cheapest", "under $X" queries |
| `multi_turn` | 4 | References prior context (cart state, prior search) |
| `escalate` | 2 | Explicit human request |
| `refund` | 2 | Return/refund intent |
| `greeting` | 2 | Small talk |

## Relevance scoring

Each relevant product is scored 1-3:
- **3** = exactly what the user wanted
- **2** = relevant alternative
- **1** = tangentially related

Empty `relevant` array means: the system should not return any products for this query
(e.g. greetings, escalations) OR the query is intentionally hard (no matches expected).
