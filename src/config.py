"""Centralized configuration. Single source of truth for all env-driven knobs."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5", alias="ANTHROPIC_MODEL")

    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    embedding_backend: Literal["local", "voyage"] = Field(default="local", alias="EMBEDDING_BACKEND")

    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    rerank_backend: Literal["local", "cohere"] = Field(default="local", alias="RERANK_BACKEND")

    database_url: str = Field(
        default="postgresql://ecommerce:ecommerce_dev@localhost:5432/ecommerce",
        alias="DATABASE_URL",
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    vector_store: Literal["qdrant", "pgvector", "chroma"] = Field(default="qdrant", alias="VECTOR_STORE")

    max_retrieval_k: int = Field(default=20, alias="MAX_RETRIEVAL_K")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    escalation_refund_threshold_usd: float = Field(default=100.0, alias="ESCALATION_REFUND_THRESHOLD_USD")

    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
