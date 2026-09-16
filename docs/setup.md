# SETU — Setup

## Prerequisites
- macOS/Linux, Python managed by `uv` (project pins 3.12 — uv fetches it automatically)
- [uv](https://docs.astral.sh/uv/) installed (`brew install uv`)
- [Ollama](https://ollama.com) installed (LLM runtime — needed from the understanding/reasoning phases onward)
- ffmpeg (needed only for the speech phases)

## Install

```bash
make setup        # uv venv --python 3.12 && uv sync --extra dev
cp .env.example .env
```

## Verify

```bash
make test         # pytest
make lint         # ruff
```

## What works right now
Phase 0 state: repository skeleton, dependency management, tests, lint.
No AI functionality exists yet — see `docs/SETU_MASTER_IMPLEMENTATION_PLAN.md`
for the full phase plan.
