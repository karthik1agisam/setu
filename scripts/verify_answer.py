#!/usr/bin/env python3
"""Verify a single answer's claims against retrieved evidence.

Usage:  uv run python scripts/verify_answer.py "answer text" --scheme pmkisan
"""

from __future__ import annotations

import argparse
import json
import sys

from ai.reasoning.llm import is_available
from ai.retrieval.bm25 import BM25Index
from ai.retrieval.store import ChunkStore
from ai.verification.claims import extract_claims
from ai.verification.verify import verify_claims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answer")
    ap.add_argument("--question", default="")
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if not is_available():
        print("Ollama not running")
        return 1

    store = ChunkStore()
    bm25 = BM25Index(store)
    query = args.question or args.answer
    hits = bm25.search(query, k=args.k, scheme=args.scheme)
    ev = [store.get(cid) for cid, _ in hits]

    cl = extract_claims(args.answer)
    vr = verify_claims(cl.claims, ev)
    out = {
        "n_claims": len(vr.claims),
        "unsupported_rate": vr.unsupported_rate,
        "claims": [vars(c) for c in vr.claims],
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
