"""Download the Instacart Market Basket Analysis dataset and verify it with SHA-256.

Source: https://www.kaggle.com/competitions/instacart-market-basket-analysis/data
License: free for non-commercial / academic use (Kaggle competition rules).

Two ways to obtain the files:

1. Kaggle CLI (``pip install kaggle`` + ``~/.kaggle/kaggle.json``)::

       python scripts/download_data.py

2. A mirror URL you control (a zip with products.csv, aisles.csv, departments.csv)::

       python scripts/download_data.py --url https://example.org/instacart.zip

After extraction, the SHA-256 of every CSV is appended to ``data/MANIFEST.txt``.
Then run ``python -m src.ingestion.preprocess`` to build ``data/processed/products.csv``.
Without credentials or a URL the script exits 0 and the project keeps using the
synthetic catalog. Network access is mocked in the tests.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.txt"
COMPETITION = "instacart-market-basket-analysis"
EXPECTED_FILES = ("products.csv", "aisles.csv", "departments.csv")


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_manifest(paths: Sequence[Path], manifest: Path = MANIFEST) -> list[str]:
    """Append ``sha256  path`` lines for ``paths`` (replacing earlier lines for the same path)."""
    existing = manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []
    lines = list(existing)
    for p in paths:
        rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        lines = [ln for ln in lines if f"  {rel}" not in ln]
        lines.append(f"{sha256_of(p)}  {rel}  source=instacart-raw")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def extract_all(raw_dir: Path) -> list[Path]:
    """Unzip every ``*.zip`` in ``raw_dir`` (Kaggle ships nested zips) and return the CSVs."""
    for _ in range(2):  # outer competition zip, then per-file zips
        for z in sorted(raw_dir.glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw_dir)
            z.unlink()
    return [raw_dir / name for name in EXPECTED_FILES if (raw_dir / name).exists()]


def download_with_kaggle(raw_dir: Path) -> int:
    """Run the Kaggle CLI; returns its exit code (2 when the CLI is missing)."""
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(raw_dir)],
            check=True,
            timeout=1800,
        )
    except FileNotFoundError:
        print("kaggle CLI not on PATH. Install with: pip install kaggle", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"kaggle returned {exc.returncode}", file=sys.stderr)
        return exc.returncode
    return 0


def download_with_url(url: str, raw_dir: Path, timeout: float = 120.0) -> Path:
    """Stream a zip from ``url`` into ``raw_dir``."""
    import httpx

    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "instacart.zip"
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        with target.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="mirror zip URL instead of the Kaggle CLI")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args(argv)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        download_with_url(args.url, raw_dir)
    elif os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        code = download_with_kaggle(raw_dir)
        if code != 0:
            return code
    else:
        print(
            "No Kaggle credentials (~/.kaggle/kaggle.json) and no --url: keeping the synthetic catalog."
        )
        return 0

    csvs = extract_all(raw_dir)
    missing = set(EXPECTED_FILES) - {p.name for p in csvs}
    if missing:
        print(f"download finished but files are missing: {sorted(missing)}", file=sys.stderr)
        return 1
    record_manifest(csvs, Path(args.manifest))
    print(f"downloaded {len(csvs)} files into {raw_dir}; hashes appended to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
