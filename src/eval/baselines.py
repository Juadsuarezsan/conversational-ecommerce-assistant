"""Baselines for the retrieval eval.

* **No-AI baseline**: BM25 only (already an ablation row; reported as the baseline).
* **Zero-shot LLM baseline**: Claude names products from memory with no retrieval;
  its answers are matched back to catalog ids by name. Needs ``ANTHROPIC_API_KEY``;
  the client is patched in tests and the cell stays *pending* otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import SecretStr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.eval.metrics import aggregate
from src.eval.runner import relevance_map, score_ranking
from src.retrieval.bm25 import BM25Index

ZERO_SHOT_PROMPT = (
    "You are a grocery search engine with no access to a catalog. For the user query, list "
    "up to 10 product names you would expect a US online supermarket to sell, most relevant "
    'first. Return JSON only: {"products": ["name", ...]}'
)


class BaselineUnavailable(RuntimeError):
    """Raised when the zero-shot baseline cannot run (no API key)."""


def match_names_to_catalog(names: Sequence[str], bm25: BM25Index) -> list[int]:
    """Map free-text product names to catalog ids with BM25 (top-1 per name, de-duplicated)."""
    ids: list[int] = []
    for name in names:
        hits = bm25.search(name, k=1)
        if hits and hits[0].product_id not in ids:
            ids.append(hits[0].product_id)
    return ids


class ZeroShotBaseline:
    """Claude without retrieval; ``_build_chat`` is patched in tests."""

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 20.0):
        s = get_settings()
        self.model = model or s.anthropic_model
        self.api_key = api_key if api_key is not None else s.anthropic_api_key
        self.timeout = timeout

    def _build_chat(self) -> Any:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.model,
            api_key=SecretStr(self.api_key or ""),
            temperature=0.0,
            max_tokens=400,
            timeout=self.timeout,
            max_retries=0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((ValueError, TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def suggest(self, query: str) -> list[str]:
        """Ask the model for product names (no catalog)."""
        if not self.api_key:
            raise BaselineUnavailable("zero-shot baseline requires ANTHROPIC_API_KEY")
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = await self._build_chat().ainvoke(
            [SystemMessage(content=ZERO_SHOT_PROMPT), HumanMessage(content=query)]
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            data = json.loads(text)
            return [str(n) for n in data.get("products", [])][:10]
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ValueError("zero-shot baseline returned invalid JSON") from exc


async def evaluate_zero_shot(
    records: Sequence[dict[str, Any]], bm25: BM25Index, baseline: ZeroShotBaseline | None = None
) -> dict[str, Any]:
    """Score the zero-shot baseline with the same four metrics as the ablation."""
    zs = baseline or ZeroShotBaseline()
    scores: list[dict[str, float]] = []
    for rec in records:
        names = await zs.suggest(rec["query"])
        ids = match_names_to_catalog(names, bm25)
        scores.append(score_ranking(ids, relevance_map(rec)))
    return {"n_queries": len(scores), "overall": aggregate(scores)}
