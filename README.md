# SETU

**Verified Multilingual Voice AI for Welfare-Scheme Eligibility**

SETU answers one question — *"am I eligible for this government welfare
scheme?"* — in English, Telugu, or Hindi, by voice or text. Its defining
property: **every eligibility claim is verified against retrieved official
government clauses before it is shown.** If evidence can't support an answer,
SETU says so — it never invents rules.

## Status

Phase 1 — document acquisition complete: official guideline PDFs for 4
schemes (PM-KISAN, PM-JAY, PMAY-G, PMS-SC) downloaded with provenance
manifest (`data/raw/manifest.csv`). No AI functionality yet; development
proceeds in verified phases per
[`docs/SETU_MASTER_IMPLEMENTATION_PLAN.md`](docs/SETU_MASTER_IMPLEMENTATION_PLAN.md).

```bash
uv run python scripts/download_docs.py   # fetch + validate corpus
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
