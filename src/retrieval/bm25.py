"""BM25 keyword retrieval over the product catalog. Built in-memory for 50K products."""
from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from src.api.schemas import RetrievedProduct

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 over (product_name + aisle + department)."""

    def __init__(self, products: Sequence[dict]) -> None:
        self.products = list(products)
        corpus: list[list[str]] = []
        for p in self.products:
            text = f"{p['product_name']} {p.get('aisle', '')} {p.get('department', '')}"
            corpus.append(tokenize(text))
        self.bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        self._by_id = {p["product_id"]: p for p in self.products}

    def search(self, query: str, k: int = 10) -> list[RetrievedProduct]:
        if not query.strip():
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out: list[RetrievedProduct] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            p = self.products[idx]
            out.append(RetrievedProduct(
                product_id=p["product_id"],
                product_name=p["product_name"],
                aisle=p.get("aisle", ""),
                department=p.get("department", ""),
                price_usd=p.get("price_usd"),
                avg_rating=p.get("avg_rating"),
                rating_count=p.get("rating_count", 0),
                in_stock=p.get("in_stock", True),
                score=float(score),
                source="bm25",
            ))
        return out

    def __len__(self) -> int:
        return len(self.products)
