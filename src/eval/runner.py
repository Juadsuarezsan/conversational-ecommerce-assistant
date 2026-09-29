"""Retrieval ablation over ``data/eval/queries.jsonl``.

For every retrieval query (``lookup`` / ``semantic`` / ``comparative``) four
strategies are scored separately: BM25 only, dense only, hybrid (RRF) and
hybrid + rerank. Multi-turn queries are evaluated by
:mod:`src.eval.conversation` through the full agent instead.

The report says which embedding and reranker produced the numbers. When they are
the deterministic fallbacks (hash embeddings / lexical reranker) the report is
labelled accordingly and must not be presented as the production system.

CLI::

    python -m src.eval.runner --json
    python -m src.eval.runner --vector-store chroma
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import VectorStoreKind, get_settings
from src.eval.metrics import aggregate, mrr, ndcg_at_k, percentile, precision_at_k, recall_at_k
from src.ingestion.synthetic_catalog import build_catalog
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseStore, build_store
from src.retrieval.embeddings import EmbeddingProvider, get_embedding_provider
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import Reranker, get_reranker
from src.retrieval.structured import apply_filter, lexically_related, parse_filters

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "queries.jsonl"
RETRIEVAL_CATEGORIES = ("lookup", "semantic", "comparative")
STRATEGIES = ("bm25", "dense", "hybrid", "rerank")
FALLBACK_EMBEDDINGS = {"hash"}
FALLBACK_RERANKERS = {"lexical"}


def load_eval_set(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    """Read the JSONL eval set."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def relevance_map(query: dict[str, Any]) -> dict[int, int]:
    """``{product_id: relevance}`` for one eval record."""
    return {int(r["product_id"]): int(r["relevance"]) for r in query.get("relevant", [])}


def score_ranking(ids: Sequence[int], rel: dict[int, int]) -> dict[str, float]:
    """The four mandatory retrieval metrics for one ranking."""
    return {
        "p_at_5": precision_at_k(ids, rel, 5),
        "r_at_10": recall_at_k(ids, rel, 10),
        "mrr": mrr(ids, rel),
        "ndcg_10": ndcg_at_k(ids, rel, 10),
    }


def provenance(embedder: EmbeddingProvider, reranker: Reranker) -> dict[str, Any]:
    """Describe which components produced a run and whether they are fallbacks."""
    is_fallback = embedder.name in FALLBACK_EMBEDDINGS or reranker.name in FALLBACK_RERANKERS
    return {
        "embedding": embedder.name,
        "reranker": reranker.name,
        "deterministic_fallback": is_fallback,
        "label": (
            "fallback determinista (embedding por hash / rerank lexico), sin modelo"
            if is_fallback
            else "modelos de produccion"
        ),
    }


async def evaluate(
    vector_store: VectorStoreKind = "in_memory",
    *,
    eval_set: Sequence[dict[str, Any]] | None = None,
    catalog: Sequence[dict[str, Any]] | None = None,
    embedder: EmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    k: int = 20,
    rerank_top_n: int = 10,
) -> dict[str, Any]:
    """Run the four-strategy ablation and return a JSON-serialisable report.

    Args:
        vector_store: Dense store to benchmark.
        eval_set: Queries; defaults to ``data/eval/queries.jsonl`` (retrieval categories only).
        catalog: Products; defaults to the synthetic catalog.
        embedder: Injected embedding provider (defaults to settings).
        reranker: Injected reranker (defaults to settings).
        k: Candidates per strategy before reranking.
        rerank_top_n: Candidates kept after reranking.
    """
    get_settings()
    queries = [q for q in (eval_set or load_eval_set()) if q["category"] in RETRIEVAL_CATEGORIES]
    products = list(catalog) if catalog is not None else build_catalog()
    emb = embedder or get_embedding_provider()
    rr = reranker or get_reranker()

    bm25 = BM25Index(products)
    dense: DenseStore = build_store(vector_store, embedder=emb, collection="products_eval")
    t_index = time.perf_counter()
    indexed = await dense.index(products)
    index_ms = (time.perf_counter() - t_index) * 1000
    logger.info("Indexed {} products into {} in {:.0f} ms", indexed, dense.name, index_ms)
    retriever = HybridRetriever(bm25=bm25, dense=dense, embedder=emb)

    per_strategy: dict[str, list[dict[str, float]]] = {s: [] for s in STRATEGIES}
    latencies: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    by_category: dict[str, dict[str, list[dict[str, float]]]] = {}
    per_query: list[dict[str, Any]] = []

    for query in queries:
        q = str(query["query"])
        rel = relevance_map(query)
        cat = str(query["category"])
        by_category.setdefault(cat, {s: [] for s in STRATEGIES})

        t0 = time.perf_counter()
        bm25_hits = bm25.search(q, k=k)
        latencies["bm25"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        q_emb = await emb.embed_query(q)
        dense_hits = await dense.search(q_emb, k=k)
        latencies["dense"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        hybrid_hits = await retriever.search(q, k=k)
        latencies["hybrid"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        reranked = await rr.rerank(q, hybrid_hits, top_n=rerank_top_n)
        f = parse_filters(q)
        if not f.is_empty():
            filtered = apply_filter(lexically_related(q, reranked), f) or apply_filter(
                lexically_related(q, hybrid_hits), f
            )
            if filtered:
                reranked = filtered[:rerank_top_n]
        latencies["rerank"].append((time.perf_counter() - t0) * 1000)

        rankings = {
            "bm25": [r.product_id for r in bm25_hits],
            "dense": [r.product_id for r in dense_hits],
            "hybrid": [r.product_id for r in hybrid_hits],
            "rerank": [r.product_id for r in reranked],
        }
        row: dict[str, Any] = {"id": query["id"], "category": cat, "query": q}
        for strategy, ids in rankings.items():
            scores = score_ranking(ids, rel)
            per_strategy[strategy].append(scores)
            by_category[cat][strategy].append(scores)
            row[strategy] = {**scores, "top": ids[:5]}
        per_query.append(row)

    await dense.close()

    latency_summary = {
        s: {
            "p50_ms": percentile(latencies[s], 50),
            "p95_ms": percentile(latencies[s], 95),
            "mean_ms": statistics.mean(latencies[s]) if latencies[s] else 0.0,
        }
        for s in STRATEGIES
    }
    return {
        "n_queries": len(queries),
        "n_products": len(products),
        "vector_store": dense.name,
        "index_ms": round(index_ms, 1),
        "provenance": provenance(emb, rr),
        "overall": {s: aggregate(per_strategy[s]) for s in STRATEGIES},
        "latency": latency_summary,
        "per_category": {
            cat: {s: aggregate(by_category[cat][s]) for s in STRATEGIES} for cat in by_category
        },
        "per_query": per_query,
    }


def format_report(report: dict[str, Any]) -> str:
    """Human-readable table of a report."""
    sep = "=" * 78
    lines = [
        sep,
        f"E-COMMERCE RETRIEVAL EVAL  (store: {report['vector_store']}, n={report['n_queries']})",
        f"components: {report['provenance']['label']}",
        sep,
        f"\n{'Strategy':<10} {'P@5':>7} {'R@10':>8} {'MRR':>7} {'nDCG@10':>10} {'p95_ms':>9}",
        "-" * 78,
    ]
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exit code 1 when rerank P@5 is below ``--gate``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the raw JSON report")
    parser.add_argument(
        "--vector-store",
        default="in_memory",
        choices=["in_memory", "qdrant", "pgvector", "chroma"],
    )
    parser.add_argument("--gate", type=float, default=0.0, help="minimum rerank P@5")
    args = parser.parse_args(argv)
    report = asyncio.run(evaluate(vector_store=args.vector_store))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_report(report))
    return 1 if report["overall"]["rerank"]["p_at_5"] < args.gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
