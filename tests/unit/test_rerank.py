"""Rerankers: lexical fallback, Cohere (mocked), cross-encoder (fake module), factory."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from src.api.schemas import RetrievedProduct
from src.config import get_settings
from src.retrieval.rerank import CohereReranker, CrossEncoderReranker, LexicalReranker, get_reranker


def _rp(pid: int, name: str, score: float = 0.5, aisle: str = "cereal") -> RetrievedProduct:
    return RetrievedProduct(
        product_id=pid, product_name=name, aisle=aisle, department="breakfast", score=score
    )


CANDS = [
    _rp(1, "Dog Treats", 0.9, aisle="pets"),
    _rp(2, "Cheerios Honey Nut", 0.5),
    _rp(3, "Kids Cereal Pack", 0.4),
]


async def test_lexical_reranker_prefers_query_tokens() -> None:
    out = await LexicalReranker().rerank("kids cereal", CANDS, top_n=2)
    assert [r.product_id for r in out] == [3, 2]
    assert all(r.source == "rerank" for r in out)


async def test_lexical_reranker_edge_cases() -> None:
    rr = LexicalReranker()
    assert await rr.rerank("x", [], top_n=3) == []
    out = await rr.rerank("!!!", CANDS, top_n=2)  # no tokens → keep incoming order
    assert [r.product_id for r in out] == [1, 2]


async def test_cohere_reranker_maps_indices(mocker: Any) -> None:
    class _R:
        def __init__(self, index: int, score: float) -> None:
            self.index, self.relevance_score = index, score

    class _Res:
        results: ClassVar[list[Any]] = [_R(2, 0.9), _R(1, 0.3)]

    fake = mocker.MagicMock()
    fake.rerank = mocker.AsyncMock(return_value=_Res())
    mocker.patch("cohere.AsyncClient", return_value=fake)
    rr = CohereReranker("key")
    out = await rr.rerank("q", CANDS, top_n=2)
    assert [r.product_id for r in out] == [3, 2] and out[0].score == 0.9
    assert await rr.rerank("q", [], top_n=2) == []


async def test_cross_encoder_with_fake_module(mocker: Any) -> None:
    class _CE:
        def __init__(self, name: str) -> None:
            pass

        def predict(self, pairs: Any, show_progress_bar: bool = False) -> Any:
            import numpy as np

            return np.array([0.1, 0.9, 0.5])

    module = mocker.MagicMock()
    module.CrossEncoder = _CE
    mocker.patch.dict("sys.modules", {"sentence_transformers": module})
    rr = CrossEncoderReranker("cross-encoder/x")
    out = await rr.rerank("q", CANDS, top_n=2)
    assert [r.product_id for r in out] == [2, 3]
    assert await rr.rerank("q", [], top_n=2) == []


def test_factory_default_and_cohere_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_reranker().name == "lexical"
    monkeypatch.setenv("RERANK_BACKEND", "cohere")
    get_settings.cache_clear()
    get_reranker.cache_clear()
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        get_reranker()


def test_factory_local_without_extra(monkeypatch: pytest.MonkeyPatch, mocker: Any) -> None:
    monkeypatch.setenv("RERANK_BACKEND", "local")
    get_settings.cache_clear()
    get_reranker.cache_clear()
    mocker.patch.dict("sys.modules", {"sentence_transformers": None})
    with pytest.raises(RuntimeError, match="ml"):
        get_reranker()
