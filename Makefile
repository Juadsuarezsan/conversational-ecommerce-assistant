.PHONY: install install-ml lint format typecheck test eval preprocess run demo-bake gitleaks clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-ml:
	$(PIP) install -e ".[dev,ml]"

lint:
	.venv/bin/ruff check .
	.venv/bin/black --check .

format:
	.venv/bin/ruff check . --fix
	.venv/bin/black .

typecheck:
	.venv/bin/mypy --strict src/

test:
	$(PY) -m pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70

eval:
	$(PY) -m eval.run --stores in_memory,chroma

preprocess:
	$(PY) -m src.ingestion.preprocess

run:
	.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

demo-bake:
	$(PY) scripts/bake_demo.py

gitleaks:
	gitleaks detect --no-banner --redact --source .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov coverage.xml data/cache
