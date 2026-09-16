# ADR-001: Pin Python to 3.12

**Status:** Accepted · **Date:** 2026-09-17

## Context
System Python on the dev machine is 3.14.7 (Homebrew). Several ML dependencies
the project will need (PyTorch, CTranslate2/faster-whisper, FlagEmbedding) have
spotty or absent wheel support for 3.14 at project start.

## Decision
Pin `requires-python = ">=3.12,<3.13"` and create the project venv with
`uv venv --python 3.12` (uv fetches a standalone 3.12 — no pyenv needed).

## Consequences
- Reproducible: any contributor gets the same interpreter via `make setup`.
- Constraint documented: when 3.13/3.14 wheel coverage matures, revisit.
