#!/usr/bin/env python3
"""Run the Understand stage on a question.

Usage:  uv run python scripts/understand_query.py "I am a farmer with 2 acres, do I get PM-KISAN?"
"""

from __future__ import annotations

import sys

from ai.reasoning.llm import LLMUnavailableError, is_available
from ai.reasoning.understand import understand


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    if not is_available():
        print("Ollama not running — `ollama serve` and `ollama pull qwen2.5:7b-instruct`")
        return 1
    q = " ".join(sys.argv[1:])
    try:
        res = understand(q)
    except LLMUnavailableError as e:
        print(e)
        return 1
    print(res.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
