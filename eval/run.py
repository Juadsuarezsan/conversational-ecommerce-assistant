"""Single command that regenerates every number in ``eval/RESULTS.md``.

    python -m eval.run                      # in-memory + chroma, hash embeddings
    python -m eval.run --stores in_memory   # faster
    python -m eval.run --stores qdrant,pgvector,chroma   # needs docker compose up

Writes ``eval/runs/<date>-<name>.json`` and rewrites ``eval/RESULTS.md``. Cells that
need an API key or a service that is not reachable are written as *pendiente*.
Runs made with the deterministic fallbacks (hash embeddings, lexical reranker,
heuristic router) are labelled so and are never presented as production numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from src.agents.orchestrator import build_graph
from src.agents.runtime import build_runtime
from src.config import get_settings
from src.eval.baselines import BaselineUnavailable, evaluate_zero_shot
from src.eval.conversation import evaluate_conversations
from src.eval.judge import JudgeUnavailable, judge_conversation
from src.eval.ragas_eval import RagasUnavailable, run_ragas
from src.eval.runner import RETRIEVAL_CATEGORIES, evaluate, load_eval_set

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
RESULTS_MD = ROOT / "RESULTS.md"
ALL_STORES = ("in_memory", "qdrant", "pgvector", "chroma")
PENDING_KEY = "pendiente (requiere ANTHROPIC_API_KEY)"
PENDING_DOCKER = "pendiente (requiere Docker: docker compose up)"
#: Public list prices used for the cost column (USD per 1K queries), stated as assumptions.
COST_PER_1K: dict[str, str] = {
    "in_memory": "$0 (proceso local)",
    "chroma": "$0 (embebido, disco local)",
    "qdrant": "$0 self-hosted (contenedor local)",
    "pgvector": "$0 self-hosted (contenedor local)",
}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT.parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


async def _retrieval_runs(stores: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for store in stores:
        try:
            out[store] = await evaluate(vector_store=store)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("store {} not evaluated: {}", store, exc)
            out[store] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


async def _llm_gated(records: list[dict[str, Any]], rt: Any) -> dict[str, Any]:
    """Zero-shot baseline, RAGAS and LLM judge: run only when a key exists."""
    gated: dict[str, Any] = {}
    retrieval = [r for r in records if r["category"] in RETRIEVAL_CATEGORIES]
    try:
        gated["zero_shot"] = await evaluate_zero_shot(retrieval, rt.bm25)
    except BaselineUnavailable as exc:
        gated["zero_shot"] = {"pending": str(exc)}
    try:
        gated["ragas"] = await run_ragas(retrieval, rt)
    except RagasUnavailable as exc:
        gated["ragas"] = {"pending": str(exc)}
    return gated


async def _judge(conversations: dict[str, Any]) -> dict[str, Any]:
    verdicts = []
    try:
        for outcome in conversations["conversations"]:
            v = await judge_conversation(outcome)
            verdicts.append({"id": outcome["id"], "scores": v.scores, "rationale": v.rationale})
    except JudgeUnavailable as exc:
        return {"pending": str(exc)}
    n = len(verdicts)
    return {
        "n": n,
        "task_completion_rate": sum(v["scores"].get("1", 0) for v in verdicts) / n if n else 0.0,
        "mean_score": sum(sum(v["scores"].values()) for v in verdicts) / n if n else 0.0,
        "verdicts": verdicts,
    }


async def run_all(stores: list[str]) -> dict[str, Any]:
    """Execute every evaluation and return one JSON document."""
    settings = get_settings()
    records = load_eval_set()
    retrieval = await _retrieval_runs(stores)
    rt = await build_runtime(settings)
    rt.graph = build_graph(rt)
    conversations = await evaluate_conversations(records, rt)
    gated = await _llm_gated(records, rt)
    judge = await _judge(conversations)
    await rt.close()
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "model": settings.anthropic_model,
        "llm_enabled": settings.llm_enabled,
        "embedding_backend": settings.embedding_backend,
        "rerank_backend": settings.rerank_backend,
        "n_queries": len(records),
        "retrieval": retrieval,
        "conversations": conversations,
        "zero_shot": gated["zero_shot"],
        "ragas": gated["ragas"],
        "judge": judge,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _f(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _worst_cases(report: dict[str, Any], n: int = 10) -> list[dict[str, Any]]:
    rows = report.get("per_query", [])
    return sorted(rows, key=lambda r: (r["rerank"]["ndcg_10"], r["rerank"]["mrr"]))[:n]


def render_results(doc: dict[str, Any], stores: list[str]) -> str:
    """Render ``eval/RESULTS.md`` from a run document."""
    ret = doc["retrieval"]
    primary = next(
        (ret[s] for s in ("in_memory", "chroma", "qdrant") if "overall" in ret.get(s, {})), None
    )
    label = primary["provenance"]["label"] if primary else "sin corrida"
    lines: list[str] = [
        "# Resultados de evaluación",
        "",
        f"Generado por `python -m eval.run` el {doc['generated_at']} (commit `{doc['git_sha']}`, Python {doc['python']}).",
        "",
        "> **Etiqueta de la corrida:** "
        + (
            "**modelos de producción**"
            if not (primary and primary["provenance"]["deterministic_fallback"])
            else f"**{label}**. Embeddings `{doc['embedding_backend']}` y reranker `{doc['rerank_backend']}` "
            "son fallbacks deterministas sin modelo neuronal; las filas *dense* y *rerank* miden ese fallback, "
            "**no** el sistema de producción (voyage-3 + Cohere Rerank v3 o MiniLM + cross-encoder)."
        ),
        ">",
        "> La ground truth está escrita a mano sobre el **catálogo sintético** (213 productos, seed 20260516), "
        "no sobre Instacart real (ver `data/eval/README.md`).",
        "",
        "## Tabla obligatoria: vector DBs",
        "",
        "| Vector DB | Precision@5 | Recall@10 | nDCG@10 | Latencia p95 (dense, ms) | Costo/1K queries |",
        "|---|---|---|---|---|---|",
    ]
    for store in ("qdrant", "pgvector", "chroma", "in_memory"):
        r = ret.get(store)
        if r and "overall" in r:
            d = r["overall"]["dense"]
            lines.append(
                f"| {store} | {_f(d['p_at_5'])} | {_f(d['r_at_10'])} | {_f(d['ndcg_10'])} | "
                f"{r['latency']['dense']['p95_ms']:.1f} | {COST_PER_1K[store]} |"
            )
        else:
            reason = PENDING_DOCKER if store in ("qdrant", "pgvector") else "pendiente"
            lines.append(
                f"| {store} | {reason} | {reason} | {reason} | {reason} | {COST_PER_1K[store]} |"
            )
    lines += [
        "",
        "Las filas medidas usan la estrategia *dense* (solo vector) del store indicado, con las mismas "
        "embeddings; p95 incluye embed de la query + búsqueda. Costo/1K = infraestructura self-hosted; "
        "con `voyage-3` se suman ~$0.12/M tokens de embedding de queries (≈$0.002/1K queries de 15 tokens).",
        "",
    ]
    if primary:
        lines += [
            f"## Ablación de retrieval (store `{primary['vector_store']}`, n={primary['n_queries']} queries)",
            "",
            "| Estrategia | P@5 | R@10 | MRR | nDCG@10 | p50 ms | p95 ms |",
            "|---|---|---|---|---|---|---|",
        ]
        names = {
            "bm25": "BM25 solo (baseline sin IA)",
            "dense": "Dense solo",
            "hybrid": "Híbrido BM25 + dense (RRF k=60)",
            "rerank": "Híbrido + rerank + filtros",
        }
        for s in ("bm25", "dense", "hybrid", "rerank"):
            o = primary["overall"][s]
            lat = primary["latency"][s]
            lines.append(
                f"| {names[s]} | {_f(o['p_at_5'])} | {_f(o['r_at_10'])} | {_f(o['mrr'])} | {_f(o['ndcg_10'])} | "
                f"{lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} |"
            )
        zs = doc.get("zero_shot", {})
        if "overall" in zs:
            o = zs["overall"]
            lines.append(
                f"| Zero-shot Claude sin retrieval | {_f(o['p_at_5'])} | {_f(o['r_at_10'])} | {_f(o['mrr'])} | {_f(o['ndcg_10'])} | — | — |"
            )
        else:
            lines.append(
                f"| Zero-shot Claude sin retrieval | {PENDING_KEY} | {PENDING_KEY} | {PENDING_KEY} | {PENDING_KEY} | — | — |"
            )
        lines += [
            "",
            "### Por categoría (nDCG@10)",
            "",
            "| Categoría | BM25 | Dense | Híbrido | + rerank |",
            "|---|---|---|---|---|",
        ]
        for cat, by in primary["per_category"].items():
            lines.append(
                f"| {cat} | {_f(by['bm25']['ndcg_10'])} | {_f(by['dense']['ndcg_10'])} | "
                f"{_f(by['hybrid']['ndcg_10'])} | {_f(by['rerank']['ndcg_10'])} |"
            )
        lines.append("")
    conv = doc["conversations"]
    lines += [
        "## Conversacional (20 conversaciones multi-turn guionizadas)",
        "",
        f"Etiqueta: **{conv['label']}**.",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Task completion rate (verificación determinista de carrito/intención) | {conv['task_completion_rate']:.2f} ({int(round(conv['task_completion_rate'] * conv['n_conversations']))}/{conv['n_conversations']}) |",
        f"| Turnos promedio por tarea | {conv['avg_turns']:.2f} |",
        f"| Latencia p50 / p95 por conversación (ms) | {conv['latency_p50_ms']:.1f} / {conv['latency_p95_ms']:.1f} |",
        f"| Costo promedio por conversación (USD) | {conv['avg_cost_usd']:.4f}"
        + (" (sin LLM: $0) |" if not conv["llm_enabled"] else " |"),
    ]
    judge = doc.get("judge", {})
    if "task_completion_rate" in judge:
        lines.append(
            f"| Task completion (LLM-as-judge, criterio 1) | {judge['task_completion_rate']:.2f} |"
        )
        lines.append(f"| Puntaje medio del juez (0-5) | {judge['mean_score']:.2f} |")
    else:
        lines.append(f"| Task completion (LLM-as-judge, 50 conversaciones) | {PENDING_KEY} |")
    failed = [c for c in conv["conversations"] if not c["passed"]]
    if failed:
        lines += ["", "Conversaciones fallidas:", ""]
        for c in failed:
            lines.append(f"- `{c['id']}` \"{c['query']}\": fallan {', '.join(c['failed_checks'])}")
    lines += [
        "",
        "## RAGAS",
        "",
        "| Faithfulness | Answer Relevance | Context Precision | Context Recall |",
        "|---|---|---|---|",
    ]
    ragas = doc.get("ragas", {})
    if "faithfulness" in ragas:
        lines.append(
            f"| {_f(ragas['faithfulness'])} | {_f(ragas['answer_relevancy'])} | {_f(ragas['context_precision'])} | {_f(ragas['context_recall'])} |"
        )
    else:
        lines.append(f"| {PENDING_KEY} | {PENDING_KEY} | {PENDING_KEY} | {PENDING_KEY} |")
    if primary:
        lines += [
            "",
            "## Los 10 peores casos (híbrido + rerank, por nDCG@10)",
            "",
            "| id | query | nDCG@10 | MRR | top-5 devuelto |",
            "|---|---|---|---|---|",
        ]
        for r in _worst_cases(primary):
            lines.append(
                f"| {r['id']} | {r['query']} | {_f(r['rerank']['ndcg_10'])} | {_f(r['rerank']['mrr'])} | {', '.join(str(x) for x in r['rerank']['top'])} |"
            )
        lines += ["", "El análisis de causas está en `docs/error_analysis.md`."]
    lines += [
        "",
        "## Reproducir",
        "",
        "```bash",
        "python -m eval.run --stores " + ",".join(stores),
        "```",
        "",
        "Cada corrida queda en `eval/runs/` y este archivo se regenera por completo.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stores", default="in_memory,chroma", help="comma-separated vector stores"
    )
    parser.add_argument("--name", default="full-eval", help="suffix of the run file")
    parser.add_argument("--out-dir", default=str(RUNS_DIR))
    parser.add_argument("--results", default=str(RESULTS_MD))
    args = parser.parse_args(argv)
    stores = [s.strip() for s in args.stores.split(",") if s.strip() in ALL_STORES]
    doc = asyncio.run(run_all(stores))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    run_path = out_dir / f"{stamp}-{args.name}.json"
    run_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    Path(args.results).write_text(render_results(doc, stores), encoding="utf-8")
    logger.info("wrote {} and {}", run_path, args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
