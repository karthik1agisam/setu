# SETU

**Verified Multilingual Voice AI for Welfare-Scheme Eligibility**

SETU answers one question — *"am I eligible for this government welfare
scheme?"* — in English, Telugu, or Hindi, by voice or text. Its defining
property: **every eligibility claim is verified against retrieved official
government clauses before it is shown.** If evidence can't support an answer,
SETU says so — it never invents rules.

## Status

Phase 6 — Understand stage live: user question → structured facts via local
Qwen (Ollama), strict JSON schema + bounded repair loop, fail-closed fallback.
Retrieval stack from Phase 5 (hybrid BM25+BGE-M3, measured in
`docs/evaluation.md`).

```bash
uv run python scripts/download_docs.py   # fetch + validate corpus
make ingest && make chunk              # PDFs → chunks
uv run python scripts/build_index.py   # BGE-M3 → FAISS index
make eval-retrieval CONFIG=hybrid      # retrieval metrics
uv run python scripts/understand_query.py "your question"   # LLM extract (needs Ollama)
```

## Setup

```bash
make setup        # creates Python 3.12 venv via uv, installs deps
make test         # run tests
make lint         # ruff
```

See [`docs/setup.md`](docs/setup.md) for prerequisites and details.

## Architecture (planned)

English-core pipeline: ASR → translate → fact extraction → hybrid retrieval
(BGE-M3 + BM25 → RRF) → structured condition assessment → NLI claim
verification → deterministic eligibility aggregation → abstention gate →
localized answer + optional TTS. Every stage logged and independently
evaluated against a document-grounded benchmark.
