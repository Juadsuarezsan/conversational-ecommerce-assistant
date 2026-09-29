# Error analysis — the ten worst retrieval cases

Run: `eval/runs/2026-09-29-full-eval.json`, strategy *hybrid + rerank + filters*, in-memory
store, **deterministic fallback** (hash embeddings, lexical reranker). Ranked by nDCG@10, worst
first. Product ids refer to the synthetic catalog.

All ten are `semantic` queries. Lookup queries score nDCG@10 0.992 and comparative 0.948 with the
same stack; the semantic category scores 0.586. That split is the main finding: the fallback
retrieves by spelling, and semantic queries by definition share no words with the products they
want.

| # | id | query | nDCG@10 | returned top-5 | expected (rel 3) | cause |
|---|---|---|---|---|---|---|
| 1 | q-semantic-03 | low carb dinner options | 0.000 | 202 Dinner Rolls, 185 Soy Sauce, 165 Conditioner, 167 Body Lotion, 116 Green Beans | 10/140 cauliflower pizza, 105 chicken, 108 salmon | "low carb" appears in no product name; "dinner" matches *Dinner Rolls* literally. Needs a semantic embedding (voyage-3 / MiniLM) or a diet/nutrition attribute in the catalog. |
| 2 | q-semantic-10 | dairy free yogurt | 0.000 | 32, 6, 8 (lactose-free milks) | 36 Coconut Milk Yogurt | "free" matched the *Lactose-Free* milks through BM25; the coconut yogurt has neither "dairy" nor "free" in its name. A diet attribute (`dairy_free`) on the product row would fix it deterministically; the `StructuredFilter.diet` field already exists for it. |
| 3 | q-semantic-25 | things to make guacamole | 0.000 | 208/209 beers, 41/42 eggs, 191 Tikka Masala | 16 Avocado, 18 Lime | Zero lexical overlap; pure world knowledge (guacamole → avocado). Only an embedding trained on food text or an LLM-side query rewrite ("avocado lime onion") solves this. |
| 4 | q-semantic-30 | sandwich supplies for lunch | 0.054 | 9/97 Gluten-Free *Sandwich* Bread, 70, 158, 156 | 102 Turkey, 103 Ham, 96 Whole Wheat Bread | The literal token "sandwich" wins; deli meats have no such word. Aisle "lunchmeat" contains "lunch" but plural folding does not split compounds. |
| 5 | q-semantic-04 | snacks for a road trip | 0.234 | 3 Cheerios Snack Pack, 129 Almonds, 152 Triscuits, 128 Mixed Nuts, 127 Trail Mix | 11 granola bars, 127 trail mix | Half right: the department token "snacks" pulls the snack aisle, but the ranking inside it is arbitrary (hash similarity), so the relevance-3 items land at ranks 5-8. |
| 6 | q-semantic-22 | pantry staples for a beginner cook | 0.272 | 71-78 spices and oils | 75 olive oil, 79 spaghetti, 86 rice | Department "pantry" is matched, but "staples" carries the meaning and nothing in the catalog says it. |
| 7 | q-semantic-12 | high protein snacks | 0.305 | 3, 152, 129, 150, 127 | 128 nuts, 90/91 bars, 197 tofu | Same as #5 plus "protein" has no lexical anchor; a nutrition table (Open Food Facts, optional in the spec) is the deterministic fix. |
| 8 | q-semantic-21 | spices for a curry | 0.319 | 71 Cinnamon, 72 Pepper, 74 Paprika, 73 Salt, 176 | 191 Tikka Masala Sauce | MRR is 1.0 (a relevance-1 spice comes first) but the relevance-3 sauce is missing: "curry" appears nowhere in the catalog. |
| 9 | q-semantic-08 | something to make tacos tonight | 0.400 | 170 Toothpaste, 101 Naan, 99 Flour Tortillas, 111, 202 | 100 Corn Tortillas, 99 Flour Tortillas | Tortillas are found through the aisle text, but the top hit is toothpaste: hash collisions between character trigrams of "tonight" and "toothpaste whitening" put an unrelated item first. A real embedding does not have this failure mode. |
| 10 | q-semantic-06 | quick weekday breakfast | 0.431 | 137 Frozen Breakfast Burritos, 55, 52, 54, 3 | 5 kids oatmeal, 92 oatmeal packets | The department token "breakfast" does the work; instant oats are in the "instant oats" aisle and never mention breakfast. |

## Patterns

1. **No lexical anchor** (#1, #3, #7, #8): the query names a dish, a goal or a nutrient. The
   only deterministic mitigation is catalog enrichment (diet/nutrition attributes, which the
   `StructuredFilter.diet` path already consumes); otherwise it requires semantic embeddings.
2. **Literal token wins over meaning** (#2, #4, #9): BM25 and the lexical reranker reward exact
   words, so *Sandwich Bread* beats deli meat for "sandwich supplies". Reranking with a
   cross-encoder is designed for exactly this and is one `pip install -e ".[ml]"` away; it is
   *pendiente* because model downloads are blocked in the sandbox.
3. **Right aisle, arbitrary order** (#5, #6, #10): the department/aisle text gets the right
   neighbourhood, and the hash similarity shuffles it. Expected to improve the most with real
   embeddings, since the candidates are already in the fused top 20 (recall@10 is 0.867).

## What the fallback gets right

Lookup queries (brand + size) are the bread and butter of a grocery assistant and score 0.992
nDCG@10; comparative queries reach 0.948 because the filter/sort layer is deterministic and
independent of the embedding. The conversation set passes 20/20 scripted checks because cart
resolution is symbolic (ordinals, names, pids), not embedding-based.

## Next measurements

* Re-run `python -m eval.run` with `EMBEDDING_BACKEND=local RERANK_BACKEND=local` on a machine
  with Hugging Face access; the table will replace the fallback rows and this document is
  regenerated from the new worst-10.
* With `ANTHROPIC_API_KEY`: RAGAS, the zero-shot baseline and the LLM judge fill the remaining
  *pendiente* cells with no code change.
