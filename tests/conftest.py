"""Shared pytest fixtures + deterministic seeds."""
from __future__ import annotations

import random

import pytest

SEED = 20260516


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    random.seed(SEED)


@pytest.fixture
def catalog() -> list[dict]:
    from src.ingestion.synthetic_catalog import build_catalog
    return build_catalog()


@pytest.fixture
def bm25_index(catalog):
    from src.retrieval.bm25 import BM25Index
    return BM25Index(catalog)
