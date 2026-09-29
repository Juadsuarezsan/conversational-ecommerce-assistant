"""Single-command preprocessing: raw Instacart CSVs (or the synthetic catalog) → ``data/processed``.

    python -m src.ingestion.preprocess            # uses data/raw if present, else synthetic
    python -m src.ingestion.preprocess --synthetic

Writes ``data/processed/products.csv`` and appends its SHA-256 to ``data/MANIFEST.txt``.
The real Instacart files (``products.csv``, ``aisles.csv``, ``departments.csv``) have no
price/rating columns; they are enriched with the same deterministic generator used by
the synthetic catalog so the schema is identical (see ``docs/data_schema.md``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from src.ingestion.synthetic_catalog import RNG_SEED, build_catalog

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "MANIFEST.txt"
COLUMNS = (
    "product_id",
    "product_name",
    "aisle_id",
    "aisle",
    "department_id",
    "department",
    "price_usd",
    "avg_rating",
    "rating_count",
    "in_stock",
)


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def enrich(rows: Sequence[dict[str, Any]], seed: int = RNG_SEED) -> list[dict[str, Any]]:
    """Add deterministic price / rating / stock columns to raw Instacart rows."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                **r,
                "price_usd": round(rng.uniform(1.49, 24.99), 2),
                "avg_rating": round(rng.uniform(3.8, 4.9), 2),
                "rating_count": rng.randint(50, 6000),
                "in_stock": rng.random() > 0.05,
            }
        )
    return out


def load_raw_instacart(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]] | None:
    """Join ``products.csv`` + ``aisles.csv`` + ``departments.csv``; ``None`` when absent."""
    files = {name: raw_dir / f"{name}.csv" for name in ("products", "aisles", "departments")}
    if not all(p.exists() for p in files.values()):
        return None
    with files["aisles"].open(encoding="utf-8") as f:
        aisles = {int(r["aisle_id"]): r["aisle"] for r in csv.DictReader(f)}
    with files["departments"].open(encoding="utf-8") as f:
        departments = {int(r["department_id"]): r["department"] for r in csv.DictReader(f)}
    rows: list[dict[str, Any]] = []
    with files["products"].open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "product_id": int(r["product_id"]),
                    "product_name": r["product_name"],
                    "aisle_id": int(r["aisle_id"]),
                    "aisle": aisles.get(int(r["aisle_id"]), "missing"),
                    "department_id": int(r["department_id"]),
                    "department": departments.get(int(r["department_id"]), "missing"),
                }
            )
    return rows


def write_products(rows: Sequence[dict[str, Any]], out_dir: Path = PROCESSED_DIR) -> Path:
    """Write ``products.csv`` with the canonical column order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "products.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    return path


def update_manifest(path: Path, source: str, manifest: Path = MANIFEST, n_rows: int = 0) -> str:
    """Record ``path``'s SHA-256 in the manifest (replacing an earlier entry for the same file)."""
    digest = sha256_of(path)
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    line = f"{digest}  {rel}  source={source}  rows={n_rows}"
    existing = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
    kept = [ln for ln in existing if f"  {rel}  " not in ln]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join([*kept, line]) + "\n", encoding="utf-8")
    return digest


def run(
    *,
    synthetic: bool = False,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = PROCESSED_DIR,
    manifest: Path = MANIFEST,
) -> Path:
    """Build the processed catalog and update the manifest; returns the CSV path."""
    rows = None if synthetic else load_raw_instacart(raw_dir)
    if rows is None:
        source = "synthetic_catalog(seed=20260516)"
        rows = build_catalog()
        logger.info(
            "Raw Instacart files not found: using the synthetic catalog ({} rows)", len(rows)
        )
    else:
        source = "instacart-market-basket-analysis"
        rows = enrich(rows)
        logger.info("Loaded {} Instacart products", len(rows))
    path = write_products(rows, out_dir)
    digest = update_manifest(path, source, manifest, n_rows=len(rows))
    logger.info("wrote {} sha256={}", path, digest)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="ignore data/raw even if present")
    args = parser.parse_args(argv)
    run(synthetic=args.synthetic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
