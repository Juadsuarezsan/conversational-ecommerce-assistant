"""Download the real Instacart Market Basket dataset from Kaggle.

Requires `kaggle` CLI installed and `~/.kaggle/kaggle.json` with API credentials:
    pip install kaggle
    https://www.kaggle.com/docs/api

Without credentials the project uses `src/ingestion/synthetic_catalog.py` (built-in).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
COMPETITION = "instacart-market-basket-analysis"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("kaggle.json not found. Set up Kaggle API credentials first.")
        print("Falling back to synthetic catalog — that's fine for offline dev and CI.")
        return 0
    print(f"Downloading {COMPETITION} into {RAW_DIR} ...")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)],
            check=True,
        )
        subprocess.run(["unzip", "-o", str(RAW_DIR / f"{COMPETITION}.zip"), "-d", str(RAW_DIR)], check=False)
        return 0
    except FileNotFoundError:
        print("kaggle CLI not on PATH. Install with: pip install kaggle")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"kaggle returned {e.returncode}: {e}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
