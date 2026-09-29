"""download_data.py (mocked network / CLI) and the preprocessing command."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from src.ingestion import preprocess


def _load_script(repo_root: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        "download_data", Path(repo_root) / "scripts" / "download_data.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "products.csv", "product_id,product_name,aisle_id,department_id\n1,Heinz Ketchup,1,1\n"
        )
        zf.writestr("aisles.csv", "aisle_id,aisle\n1,pasta sauce\n")
        zf.writestr("departments.csv", "department_id,department\n1,pantry\n")
    return buf.getvalue()


def test_download_without_credentials_keeps_synthetic(
    repo_root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    mod = _load_script(repo_root)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert (
        mod.main(["--raw-dir", str(tmp_path / "raw"), "--manifest", str(tmp_path / "M.txt")]) == 0
    )
    assert "synthetic" in capsys.readouterr().out


def test_download_with_mocked_url(repo_root: str, tmp_path: Path, respx_mock: Any) -> None:
    mod = _load_script(repo_root)
    respx_mock.get("https://mirror.test/instacart.zip").respond(200, content=_zip_bytes())
    manifest = tmp_path / "MANIFEST.txt"
    assert (
        mod.main(
            [
                "--url",
                "https://mirror.test/instacart.zip",
                "--raw-dir",
                str(tmp_path / "raw"),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    lines = manifest.read_text().splitlines()
    assert len(lines) == 3 and all(len(ln.split()[0]) == 64 for ln in lines)
    digest = hashlib.sha256((tmp_path / "raw" / "aisles.csv").read_bytes()).hexdigest()
    assert any(ln.startswith(digest) for ln in lines)
    # re-running replaces lines instead of duplicating them
    assert (
        mod.main(
            [
                "--url",
                "https://mirror.test/instacart.zip",
                "--raw-dir",
                str(tmp_path / "raw"),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    assert len(manifest.read_text().splitlines()) == 3


def test_download_with_mocked_kaggle_cli(
    repo_root: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: Any
) -> None:
    mod = _load_script(repo_root)
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text("{}")
    monkeypatch.setenv("HOME", str(home))
    raw = tmp_path / "raw"

    def fake_run(cmd: Any, check: bool, timeout: float) -> None:
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "instacart-market-basket-analysis.zip").write_bytes(_zip_bytes())

    mocker.patch.object(subprocess, "run", side_effect=fake_run)
    assert mod.main(["--raw-dir", str(raw), "--manifest", str(tmp_path / "M.txt")]) == 0
    assert (raw / "products.csv").exists()
    mocker.patch.object(subprocess, "run", side_effect=FileNotFoundError())
    assert (
        mod.main(["--raw-dir", str(tmp_path / "raw2"), "--manifest", str(tmp_path / "M2.txt")]) == 2
    )
    mocker.patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(3, "kaggle"))
    assert (
        mod.main(["--raw-dir", str(tmp_path / "raw3"), "--manifest", str(tmp_path / "M3.txt")]) == 3
    )


def test_download_missing_files_fails(repo_root: str, tmp_path: Path, respx_mock: Any) -> None:
    mod = _load_script(repo_root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("products.csv", "x")
    respx_mock.get("https://mirror.test/bad.zip").respond(200, content=buf.getvalue())
    assert (
        mod.main(
            [
                "--url",
                "https://mirror.test/bad.zip",
                "--raw-dir",
                str(tmp_path / "raw"),
                "--manifest",
                str(tmp_path / "M.txt"),
            ]
        )
        == 1
    )


def test_preprocess_synthetic_and_raw(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.txt"
    out = preprocess.run(
        synthetic=True, raw_dir=tmp_path / "raw", out_dir=tmp_path / "processed", manifest=manifest
    )
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 213 and rows[0]["product_name"].startswith("Heinz")
    assert list(rows[0].keys()) == list(preprocess.COLUMNS)
    assert "synthetic_catalog" in manifest.read_text() and "rows=213" in manifest.read_text()
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(io.BytesIO(_zip_bytes())) as zf:
        zf.extractall(raw)
    out = preprocess.run(raw_dir=raw, out_dir=tmp_path / "processed", manifest=manifest)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert (
        rows == [dict(rows[0])]
        and rows[0]["aisle"] == "pasta sauce"
        and float(rows[0]["price_usd"]) > 0
    )
    lines = manifest.read_text().splitlines()
    assert len(lines) == 1 and "instacart" in lines[0]  # replaced, not duplicated
    assert preprocess.main(["--synthetic"]) == 0
