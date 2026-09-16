#!/usr/bin/env python3
"""Baseline RAG free-answer (no verification).

Usage:  uv run python scripts/rag_query.py "question" [--scheme pmkisan]
"""

from __future__ import annotations

import argparse
import sys

from ai.reasoning.llm import is_available
from ai.reasoning.rag import answer
from ai.retrieval.bm25 import BM25Index
from ai.retrieval.store import ChunkStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if not is_available():
        print("Ollama not running")
        return 1

    store = ChunkStore()
    bm25 = BM25Index(store)
    hits = bm25.search(args.question, k=args.k, scheme=args.scheme)
    evidence = [store.get(cid) for cid, _ in hits]

    print("ANSWER:")
    print(answer(args.question, evidence))
    print("\nEVIDENCE USED:")
    for c in evidence:
        print(f"  {c.chunk_id} p{c.page_start} — {c.source_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
