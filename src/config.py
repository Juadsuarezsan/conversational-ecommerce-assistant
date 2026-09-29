"""Centralized configuration: the single source of truth for every env-driven knob.

All values are read once through :func:`get_settings` (cached). Tests that mutate
environment variables must call ``get_settings.cache_clear()`` afterwards.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Production LLM, pinned to a dated release so results are reproducible.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"

EmbeddingBackend = Literal["hash", "local", "voyage"]
RerankBackend = Literal["lexical", "local", "cohere"]
VectorStoreKind = Literal["in_memory", "qdrant", "pgvector", "chroma"]
CartBackend = Literal["memory", "postgres"]
SessionBackend = Literal["memory", "redis"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional ``.env`` file.

    Every field has a safe offline default: with no keys, no Docker and no ML
    extras installed the API still boots and answers deterministically.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default=DEFAULT_ANTHROPIC_MODEL, alias="ANTHROPIC_MODEL")
    llm_timeout_seconds: float = Field(default=15.0, alias="LLM_TIMEOUT_SECONDS")
    #: USD per 1M input tokens for the pinned model (used for ``cost_usd``).
    llm_price_in_per_mtok: float = Field(default=3.0, alias="LLM_PRICE_IN_PER_MTOK")
    #: USD per 1M output tokens for the pinned model.
    llm_price_out_per_mtok: float = Field(default=15.0, alias="LLM_PRICE_OUT_PER_MTOK")

    # --- Embeddings / rerank ---------------------------------------------
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    voyage_model: str = Field(default="voyage-3", alias="VOYAGE_MODEL")
    embedding_backend: EmbeddingBackend = Field(default="hash", alias="EMBEDDING_BACKEND")
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="LOCAL_EMBEDDING_MODEL"
    )
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    cohere_rerank_model: str = Field(default="rerank-english-v3.0", alias="COHERE_RERANK_MODEL")
    rerank_backend: RerankBackend = Field(default="lexical", alias="RERANK_BACKEND")
    local_rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="LOCAL_RERANK_MODEL"
    )
    external_timeout_seconds: float = Field(default=10.0, alias="EXTERNAL_TIMEOUT_SECONDS")

    # --- Storage -----------------------------------------------------------
    database_url: str = Field(
        default="postgresql://ecommerce:ecommerce_dev@localhost:5432/ecommerce",
        alias="DATABASE_URL",
    )
    db_pool_min_size: int = Field(default=1, alias="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=8, alias="DB_POOL_MAX_SIZE")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    vector_store: VectorStoreKind = Field(default="in_memory", alias="VECTOR_STORE")
    cart_backend: CartBackend = Field(default="memory", alias="CART_BACKEND")
    session_backend: SessionBackend = Field(default="memory", alias="SESSION_BACKEND")
    session_ttl_seconds: int = Field(default=1800, alias="SESSION_TTL_SECONDS")
    chroma_persist_dir: str = Field(default="./data/cache/chroma", alias="CHROMA_PERSIST_DIR")

    # --- Retrieval knobs -------------------------------------------------
    max_retrieval_k: int = Field(default=20, alias="MAX_RETRIEVAL_K")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    escalation_refund_threshold_usd: float = Field(
        default=100.0, alias="ESCALATION_REFUND_THRESHOLD_USD"
    )

    # --- HTTP security -----------------------------------------------------
    #: Comma-separated list of allowed origins. Never ``*`` in production.
    cors_origins: str = Field(default="http://localhost:8501", alias="CORS_ORIGINS")
    rate_limit: str = Field(default="60/minute", alias="RATE_LIMIT")

    # --- Observability -----------------------------------------------------
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="ecommerce-assistant", alias="LANGSMITH_PROJECT")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def cors_origin_list(self) -> list[str]:
        """Return CORS origins as a list, ignoring blanks."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        """Whether a real Anthropic key is configured."""
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance (cached)."""
    return Settings()
