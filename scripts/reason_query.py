#!/usr/bin/env python3
"""End-to-end English pipeline: understand → retrieve → reason → verdict.

Usage:  uv run python scripts/reason_query.py "I am a farmer..." [--scheme pmkisan]
"""

from __future__ import annotations

import argparse
import json
import sys

from ai.reasoning.aggregate import aggregate
from ai.reasoning.llm import is_available
from ai.reasoning.reason import reason
from ai.reasoning.understand import understand
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

    u = understand(args.question)
    scheme = args.scheme or (u.scheme_guess.value if u.scheme_guess != "unknown" else None)
    if not scheme:
        print(json.dumps({"verdict": "cannot_determine", "reason": "scheme unknown"}, indent=2))
        return 0

    store = ChunkStore()
    bm25 = BM25Index(store)
    hits = bm25.search(args.question, k=args.k, scheme=scheme)
    evidence = [store.get(cid) for cid, _ in hits]

    rr = reason(u, evidence)
    verdict = aggregate(rr.conditions)

    out = verdict.model_dump()
    out["scheme"] = scheme
    out["evidence_used"] = [
        {"chunk_id": c.chunk_id, "page": c.page_start, "source_url": c.source_url}
        for c in evidence
    ]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
