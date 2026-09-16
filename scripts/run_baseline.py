#!/usr/bin/env python3
"""Run the baseline RAG over the seed eval questions → save outputs for
later claim-level verification scoring (Phase 9+).

Usage:  uv run python scripts/run_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ai.reasoning.llm import is_available
from ai.reasoning.rag import answer
from ai.retrieval.bm25 import BM25Index
from ai.retrieval.store import ChunkStore

EVAL = Path("data/benchmark/retrieval_eval.jsonl")
OUT = Path("experiments/results/baseline_rag_outputs.jsonl")


def main() -> int:
    if not is_available():
        print("Ollama not running")
        return 1
    store = ChunkStore()
    bm25 = BM25Index(store)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w") as fh:
        for line in EVAL.open():
            item = json.loads(line)
            hits = bm25.search(item["question"], k=5, scheme=item["scheme"])
            ev = [store.get(cid) for cid, _ in hits]
            text = answer(item["question"], ev)
            rec = {
                "qid": item["qid"],
                "question": item["question"],
                "scheme": item["scheme"],
                "answer": text,
                "retrieved": [c.chunk_id for c in ev],
                "gold": item["gold_chunk_ids"],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{item['qid']}: {text[:90]}…")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
