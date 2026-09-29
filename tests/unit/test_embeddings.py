"""Embedding providers: hash determinism, Voyage (mocked), factory selection."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import pytest

from src.config import get_settings
from src.retrieval.embeddings import (
    EMBEDDING_DIM,
    HashEmbeddings,
    LocalEmbeddings,
    VoyageEmbeddings,
    _pad,
    get_embedding_provider,
)


async def test_hash_embeddings_are_deterministic_and_normalised(
    hash_embedder: HashEmbeddings,
) -> None:
    a = await hash_embedder.embed_query("Heinz ketchup")
    b = await hash_embedder.embed_query("Heinz ketchup")
    assert a == b and len(a) == EMBEDDING_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)
    docs = await hash_embedder.embed_documents(["a", "b"])
    assert len(docs) == 2
    assert await hash_embedder.embed_query("") == [0.0] * EMBEDDING_DIM


async def test_hash_embeddings_reflect_lexical_overlap(hash_embedder: HashEmbeddings) -> None:
    def cos(x: list[float], y: list[float]) -> float:
        return sum(a * b for a, b in zip(x, y, strict=True))

    q = await hash_embedder.embed_query("lactose free milk")
    near = await hash_embedder.embed_query("Lactose-Free Milk 1 Gallon")
    far = await hash_embedder.embed_query("Dog Treats Bacon")
    assert cos(q, near) > cos(q, far)


def test_pad_truncates_and_extends() -> None:
    assert _pad([1.0, 2.0], dim=3) == [1.0, 2.0, 0.0]
    assert _pad([1.0, 2.0, 3.0, 4.0], dim=3) == [1.0, 2.0, 3.0]


async def test_voyage_client_is_wrapped(mocker: Any) -> None:
    class _Result:
        embeddings: ClassVar[list[list[float]]] = [[0.1] * 4]

    fake = mocker.MagicMock()
    fake.embed = mocker.AsyncMock(return_value=_Result())
    mocker.patch("voyageai.AsyncClient", return_value=fake)
    v = VoyageEmbeddings("key", timeout=3.0)
    q = await v.embed_query("x")
    assert len(q) == EMBEDDING_DIM and q[:4] == [0.1] * 4
    d = await v.embed_documents(["x"])
    assert len(d) == 1 and fake.embed.await_count == 2


def test_factory_selects_hash_by_default() -> None:
    assert get_embedding_provider().name == "hash"


def test_factory_voyage_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "voyage")
    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        get_embedding_provider()


def test_factory_local_without_extra_raises(monkeypatch: pytest.MonkeyPatch, mocker: Any) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "local")
    get_settings.cache_clear()
    get_embedding_provider.cache_clear()
    mocker.patch.dict("sys.modules", {"sentence_transformers": None})
    with pytest.raises(RuntimeError, match="ml"):
        get_embedding_provider()


async def test_local_embeddings_with_fake_sentence_transformers(mocker: Any) -> None:
    class _Model:
        def encode(
            self, texts: Any, normalize_embeddings: bool = True, show_progress_bar: bool = False
        ) -> Any:
            import numpy as np

            n = 1 if isinstance(texts, str) else len(texts)
            arr = np.ones((n, 3), dtype=float)
            return arr[0] if isinstance(texts, str) else arr

    module = mocker.MagicMock()
    module.SentenceTransformer = lambda name: _Model()
    mocker.patch.dict("sys.modules", {"sentence_transformers": module})
    emb = LocalEmbeddings("fake/model")
    assert emb.name == "model"
    q = await emb.embed_query("x")
    assert len(q) == EMBEDDING_DIM and q[:3] == [1.0, 1.0, 1.0]
    assert len(await emb.embed_documents(["a", "b"])) == 2
