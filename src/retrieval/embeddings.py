"""Embedding providers behind one :class:`EmbeddingProvider` protocol.

Three backends share the same 1024-dim output so the Qdrant / pgvector schema
does not change when switching:

* ``voyage`` — Voyage AI ``voyage-3`` (paid, production).
* ``local`` — ``sentence-transformers/all-MiniLM-L6-v2`` (free, needs the ``ml`` extra;
  imported lazily so the base install never pulls torch).
* ``hash`` — deterministic hashed bag-of-words. No model, no network. It is the
  offline default used by tests and by the eval "fallback" rows, and it is
  explicitly **not** a neural embedding: results obtained with it are labelled
  as such everywhere they are reported.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Any, Protocol

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

EMBEDDING_DIM = 1024
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(Protocol):
    """Anything that turns text into fixed-size vectors."""

    dim: int
    name: str

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        ...


def _pad(vec: list[float], dim: int = EMBEDDING_DIM) -> list[float]:
    """Pad with zeros or truncate ``vec`` to exactly ``dim`` entries."""
    if len(vec) >= dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


# ---------------------------------------------------------------------------
# Deterministic hash embedding (offline default)
# ---------------------------------------------------------------------------


class HashEmbeddings:
    """Hashed bag-of-words with character trigrams, L2-normalised.

    Every token and its character trigrams are hashed (SHA-1) into one of ``dim``
    buckets with a sign bit, which makes the vector a stable lexical fingerprint.
    It captures spelling overlap ("ketchup" vs "ketchups"), not meaning; that is
    why eval rows produced with it are labelled *deterministic hash, no model*.
    """

    dim = EMBEDDING_DIM
    name = "hash"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        feats: list[str] = []
        for tok in tokens:
            feats.append(f"w:{tok}")
            padded = f"#{tok}#"
            feats.extend(f"t:{padded[i : i + 3]}" for i in range(len(padded) - 2))
        return feats

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            digest = hashlib.sha1(feat.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = 1.0 if feat.startswith("w:") else 0.5
            vec[bucket] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query string."""
        return self._vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return [self._vector(t) for t in texts]


# ---------------------------------------------------------------------------
# Voyage AI (paid)
# ---------------------------------------------------------------------------


class VoyageEmbeddings:
    """``voyage-3`` embeddings (1024-dim natively), with retries and a timeout."""

    dim = EMBEDDING_DIM
    name = "voyage-3"

    def __init__(self, api_key: str, model: str = "voyage-3", timeout: float = 10.0) -> None:
        import voyageai

        self.client: Any = voyageai.AsyncClient(  # type: ignore[attr-defined]
            api_key=api_key, timeout=timeout, max_retries=0
        )
        self.model = model
        self.name = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def embed_query(self, text: str) -> list[float]:
        """Embed a query string."""
        result = await self.client.embed([text], model=self.model, input_type="query")
        return _pad([float(x) for x in result.embeddings[0]])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        result = await self.client.embed(texts, model=self.model, input_type="document")
        return [_pad([float(x) for x in e]) for e in result.embeddings]


# ---------------------------------------------------------------------------
# Local sentence-transformers (free, optional ``ml`` extra)
# ---------------------------------------------------------------------------


class LocalEmbeddings:
    """``all-MiniLM-L6-v2`` (384-dim) padded to 1024 so the storage schema is shared.

    ``sentence_transformers`` is imported inside ``__init__`` so importing this
    module never requires torch.
    """

    dim = EMBEDDING_DIM
    name = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "EMBEDDING_BACKEND=local requires the 'ml' extra: pip install -e '.[ml]'"
            ) from exc
        logger.info("Loading local embedding model: {}", model_name)
        self.model = SentenceTransformer(model_name)
        self.name = model_name.rsplit("/", 1)[-1]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query string."""
        emb = self.model.encode(text, normalize_embeddings=True).tolist()
        return _pad([float(x) for x in emb])

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [_pad([float(x) for x in e]) for e in embs.tolist()]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Build the provider selected by ``EMBEDDING_BACKEND`` (cached per process)."""
    s = get_settings()
    if s.embedding_backend == "voyage":
        if not s.voyage_api_key:
            raise RuntimeError("EMBEDDING_BACKEND=voyage requires VOYAGE_API_KEY")
        logger.info("Embeddings: Voyage AI {}", s.voyage_model)
        return VoyageEmbeddings(
            s.voyage_api_key, model=s.voyage_model, timeout=s.external_timeout_seconds
        )
    if s.embedding_backend == "local":
        logger.info("Embeddings: local sentence-transformers")
        return LocalEmbeddings(s.local_embedding_model)
    logger.warning("Embeddings: deterministic hash (no model). Not production quality.")
    return HashEmbeddings()
