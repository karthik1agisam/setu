.PHONY: setup sync test lint format clean

setup:  ## create venv (python 3.12) and install deps
	uv venv --python 3.12
	uv sync --extra dev

sync:   ## re-sync deps after pyproject changes
	uv sync --extra dev

test:   ## run test suite
	uv run pytest -q

lint:   ## ruff lint
	uv run ruff check .

format: ## ruff format
	uv run ruff format .

clean:  ## remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__

ingest: ## PDF → structured clause blocks (data/processed/)
	uv run python scripts/ingest.py

inspect: ## inspect ingestion output
	uv run python scripts/inspect_ingestion.py

chunk:  ## blocks → retrieval chunks
	uv run python scripts/chunk_blocks.py

inspect-chunks: ## validate chunk provenance + samples
	uv run python scripts/inspect_chunks.py
