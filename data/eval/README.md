# Eval set — `queries.jsonl`

100 hand-written queries with manually assigned relevance judgements.

**Ground truth is written against the synthetic catalog**
(`src/ingestion/synthetic_catalog.py`, 213 products, seed `20260516`), not against
the real Instacart dataset. The Instacart download needs Kaggle credentials that are
not available in the development environment; when they are, the queries stay the
same and the `product_id` lists are re-labelled against the real 49,688-product
catalog (see `scripts/download_data.py`). Until then every metric in
`eval/RESULTS.md` is a measurement on the synthetic catalog and is labelled as such.

## Categories (spec sizes)

| Category | Count | What it tests |
|---|---|---|
| `lookup` | 30 | Direct product / brand / size queries ("Heinz ketchup 397 gram bottle") |
| `semantic` | 30 | Intent queries needing meaning, not keywords ("healthy breakfast for kids") |
| `comparative` | 20 | Price / rating / popularity constraints ("top rated cereals under $5") |
| `multi_turn` | 20 | Follow-ups that depend on earlier turns (cart state, prior results) |

## Record schema

```json
{"id": "q-lookup-01", "category": "lookup", "query": "Heinz ketchup 397 gram bottle",
 "relevant": [{"product_id": 1, "relevance": 3}, {"product_id": 2, "relevance": 2}]}
```

Multi-turn records add three fields:

```json
{"setup_turns": ["lactose free milk gallon"],
 "context": "User just searched lactose-free milk; 'the second one' is rank 2 of that list.",
 "expect": {"intent": "cart_op", "cart_has_rank": 2, "qty_rank": 2}}
```

* `setup_turns` are sent to the agent first, in the same session.
* `expect` is checked by `src/eval/conversation.py` after the final query:
  `intent`, `escalated`, `cart_has`, `cart_has_any`, `cart_has_rank` (rank in the
  previous result list), `qty_rank`, `cart_qty_rank`, `cart_size`, `cart_empty`,
  `response_contains`, `response_mentions_any` (`[pid:N]` citations).
  A record passes when every expectation holds. This is the deterministic
  task-completion measure; the LLM-as-judge rubric in `src/eval/judge.py` is the
  second, key-gated measure.

## Relevance scale

| Score | Meaning |
|---|---|
| 3 | exactly what the user asked for |
| 2 | a good alternative the user would accept |
| 1 | tangentially related (same aisle / complementary) |

Products not listed are treated as non-relevant. Every `product_id` is checked
against the catalog by `tests/unit/test_eval_set.py`.

## How the judgements were made

Each query was written first, then the catalog was scanned by hand for matching
products. Comparative queries were labelled by looking at the actual prices and
ratings in the synthetic catalog (for example, the only coffee under $15 is
`124 Whole Bean Colombian 12oz`). Semantic queries accept several aisles when the
intent legitimately spans them ("snacks for a road trip" includes chips, trail mix
and granola bars).
