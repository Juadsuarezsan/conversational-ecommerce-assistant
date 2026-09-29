"""BM25 keyword retrieval over the product catalog (in-memory, fine for ~50K rows)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from rank_bm25 import BM25Okapi

from src.api.schemas import RetrievalSource, RetrievedProduct

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*|[0-9]+")


def _stem(token: str) -> str:
    """Fold trivial English plurals so "cereals" matches "cereal" ("glass" is kept)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer (with plural folding) shared by BM25 and the reranker."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


def product_text(p: dict[str, Any]) -> str:
    """Text indexed for a product: name + aisle + department."""
    return f"{p['product_name']} {p.get('aisle', '')} {p.get('department', '')}"


def to_retrieved(p: dict[str, Any], score: float, source: RetrievalSource) -> RetrievedProduct:
    """Convert a catalog dict into a :class:`RetrievedProduct`."""
    return RetrievedProduct(
        product_id=int(p["product_id"]),
        product_name=str(p["product_name"]),
        aisle=str(p.get("aisle", "")),
        department=str(p.get("department", "")),
        price_usd=p.get("price_usd"),
        avg_rating=p.get("avg_rating"),
        rating_count=int(p.get("rating_count", 0) or 0),
        in_stock=bool(p.get("in_stock", True)),
        score=float(score),
        source=source,
    )


class BM25Index:
    """In-memory BM25 (k1=1.5, b=0.75) over ``product_name + aisle + department``."""

    def __init__(self, products: Sequence[dict[str, Any]]) -> None:
        self.products = list(products)
        corpus = [tokenize(product_text(p)) for p in self.products]
        self.bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        self._by_id: dict[int, dict[str, Any]] = {int(p["product_id"]): p for p in self.products}

    def search(self, query: str, k: int = 10) -> list[RetrievedProduct]:
        """Return up to ``k`` products with a positive BM25 score, best first."""
        if not query.strip():
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)[:k]
        return [
            to_retrieved(self.products[idx], float(score), "bm25")
            for idx, score in ranked
            if float(score) > 0
        ]

    def get(self, product_id: int) -> dict[str, Any] | None:
        """Look up a product by id."""
        return self._by_id.get(product_id)

    def __len__(self) -> int:
        return len(self.products)
