"""End-to-end retrieval eval over data/eval/queries.jsonl.

For each query: BM25, dense, hybrid (RRF), and hybrid+rerank are evaluated separately.
Reports Precision@5, Recall@10, MRR, nDCG@10 per strategy and per query category.

CLI:
    python -m src.eval.runner --json
    python -m src.eval.runner --vector-store qdrant
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import get_settings
from src.eval.metrics import aggregate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from src.ingestion.synthetic_catalog import build_catalog
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import ChromaStore, DenseStore, QdrantStore
from src.retrieval.embeddings import get_embedding_provider
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import get_reranker

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "eval" / "queries.jsonl"


def load_eval_set(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class InMemoryDense(DenseStore):
    """Trivial in-memory dense store — used as the offline default for the eval harness."""
    name = "in_memory"

    def __init__(self) -> None:
        self._items: list[tuple[int, list[float], dict]] = []

    async def index(self, products, batch_size: int = 256) -> int:
        embedder = get_embedding_provider()
        for i in range(0, len(products), batch_size):
            batch = list(products[i : i + batch_size])
            texts = [f"{p['product_name']} | {p.get('aisle','')} | {p.get('department','')}" for p in batch]
            embs = await embedder.embed_documents(texts)
            for p, e in zip(batch, embs):
                self._items.append((int(p["product_id"]), e, p))
        return len(self._items)

    async def search(self, query_embedding, k: int = 10):
        from src.api.schemas import RetrievedProduct
        def cos(a, b):
            num = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return num / (na * nb) if na and nb else 0.0
        scored = [(cos(query_embedding, e), pid, p) for pid, e, p in self._items]
        scored.sort(reverse=True)
        return [
            RetrievedProduct(
                product_id=pid, product_name=p["product_name"],
                aisle=p.get("aisle", ""), department=p.get("department", ""),
                price_usd=p.get("price_usd"), avg_rating=p.get("avg_rating"),
                rating_count=p.get("rating_count", 0), in_stock=p.get("in_stock", True),
                score=float(s), source="dense",
            )
            for s, pid, p in scored[:k]
        ]

    async def count(self) -> int:
        return len(self._items)

    async def close(self) -> None:
        return


async def evaluate(
    vector_store: str = "in_memory",
    only_retrieval_queries: bool = True,
) -> dict[str, Any]:
    settings = get_settings()
    eval_set = load_eval_set()
    if only_retrieval_queries:
        eval_set = [q for q in eval_set if q["category"] in ("lookup", "semantic", "comparative")]

    catalog = build_catalog()
    bm25 = BM25Index(catalog)

    dense: DenseStore
    if vector_store == "qdrant":
        dense = QdrantStore(settings.qdrant_url, collection="products_eval")
    elif vector_store == "chroma":
        dense = ChromaStore(persist_dir="./data/cache/chroma_eval")
    else:
        dense = InMemoryDense()

    logger.info(f"Indexing {len(catalog)} products into {dense.name} ...")
    indexed = await dense.index(catalog)
    logger.info(f"Indexed {indexed} into {dense.name}")

    embedder = get_embedding_provider()
    retriever = HybridRetriever(bm25=bm25, dense=dense)
    reranker = get_reranker()

    strategies = ["bm25", "dense", "hybrid", "rerank"]
    results_per_strategy: dict[str, list[dict[str, float]]] = {s: [] for s in strategies}
    latencies: dict[str, list[float]] = {s: [] for s in strategies}
    by_category: dict[str, dict[str, list[dict[str, float]]]] = {}

    for query in eval_set:
        q = query["query"]
        rel = {int(r["product_id"]): int(r["relevance"]) for r in query.get("relevant", [])}
        cat = query["category"]
        by_category.setdefault(cat, {s: [] for s in strategies})

        # BM25
        t0 = time.perf_counter()
        bm25_hits = bm25.search(q, k=20)
        latencies["bm25"].append((time.perf_counter() - t0) * 1000)
        bm25_ids = [r.product_id for r in bm25_hits]

        # Dense
        t0 = time.perf_counter()
        emb = await embedder.embed_query(q)
        dense_hits = await dense.search(emb, k=20)
        latencies["dense"].append((time.perf_counter() - t0) * 1000)
        dense_ids = [r.product_id for r in dense_hits]

        # Hybrid
        t0 = time.perf_counter()
        hybrid_hits = await retriever.search(q, k=20)
        latencies["hybrid"].append((time.perf_counter() - t0) * 1000)
        hybrid_ids = [r.product_id for r in hybrid_hits]

        # Hybrid + Rerank (top 20 → top 5)
        t0 = time.perf_counter()
        reranked = await reranker.rerank(q, hybrid_hits, top_n=10)
        latencies["rerank"].append((time.perf_counter() - t0) * 1000)
        rerank_ids = [r.product_id for r in reranked]

        for strategy, ids in [("bm25", bm25_ids), ("dense", dense_ids),
                               ("hybrid", hybrid_ids), ("rerank", rerank_ids)]:
            scores = {
                "p_at_5":  precision_at_k(ids, rel, 5),
                "r_at_10": recall_at_k(ids, rel, 10),
                "mrr":     mrr(ids, rel),
                "ndcg_10": ndcg_at_k(ids, rel, 10),
            }
            results_per_strategy[strategy].append(scores)
            by_category[cat][strategy].append(scores)

    await dense.close()

    overall = {s: aggregate(results_per_strategy[s]) for s in strategies}
    latency_summary = {
        s: {"p50_ms": _percentile(latencies[s], 50), "p95_ms": _percentile(latencies[s], 95),
            "mean_ms": statistics.mean(latencies[s]) if latencies[s] else 0.0}
        for s in strategies
    }
    per_cat = {
        cat: {s: aggregate(by_category[cat][s]) for s in strategies}
        for cat in by_category
    }

    return {
        "n_queries": len(eval_set),
        "vector_store": dense.name,
        "overall": overall,
        "latency": latency_summary,
        "per_category": per_cat,
    }


def _percentile(xs: list[float], p: int) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _format(report: dict[str, Any]) -> str:
    sep = "=" * 78
    lines = [sep, f"E-COMMERCE RETRIEVAL EVAL  (store: {report['vector_store']}, n={report['n_queries']})", sep]
    lines.append(f"\n{'Strategy':<10} {'P@5':>7} {'R@10':>8} {'MRR':>7} {'nDCG@10':>10} {'p95_ms':>9}")
    lines.append("-" * 78)
    for strategy, scores in report["overall"].items():
        lat = report["latency"][strategy]
        lines.append(
            f"{strategy:<10} {scores['p_at_5']:>7.3f} {scores['r_at_10']:>8.3f} "
            f"{scores['mrr']:>7.3f} {scores['ndcg_10']:>10.3f} {lat['p95_ms']:>9.1f}"
        )
    lines.append("\nBy category:")
    for cat, by_strat in report["per_category"].items():
        lines.append(f"  [{cat}]")
        for s, scores in by_strat.items():
            lines.append(
                f"    {s:<8} P@5={scores['p_at_5']:.2f}  R@10={scores['r_at_10']:.2f}  "
                f"MRR={scores['mrr']:.2f}  nDCG@10={scores['ndcg_10']:.2f}"
            )
    lines.append("\n" + sep)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--vector-store", default="in_memory",
                        choices=["in_memory", "qdrant", "chroma"])
    args = parser.parse_args()
    report = asyncio.run(evaluate(vector_store=args.vector_store))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_format(report))
    # Exit code 1 if rerank P@5 < 0.5 to use as a CI gate.
    if report["overall"]["rerank"]["p_at_5"] < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    main()
