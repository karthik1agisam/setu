#!/usr/bin/env python3
"""Blocks → retrieval chunks. Writes data/processed/{scheme}/*.chunks.jsonl.

Usage:  uv run python scripts/chunk_blocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from ai.ingestion.chunk import chunk_document

PROC = Path("data/processed")


def main() -> int:
    files = sorted(PROC.glob("**/*.blocks.jsonl"))
    if not files:
        print("no blocks found — run scripts/ingest.py first")
        return 1

    all_sizes = []
    for f in files:
        out = f.with_name(f.name.replace(".blocks.jsonl", ".chunks.jsonl"))
        chunks = chunk_document(f, out)
        sizes = [c.n_words for c in chunks]
        all_sizes += sizes
        print(
            f"[{f.parent.name}] {f.name}: {len(chunks)} chunks "
            f"min={min(sizes)} med={sorted(sizes)[len(sizes)//2]} max={max(sizes)} words"
        )

    all_sizes.sort()
    print(
        f"\nTOTAL {len(all_sizes)} chunks | "
        f"min={all_sizes[0]} p50={all_sizes[len(all_sizes)//2]} "
        f"p95={all_sizes[int(len(all_sizes)*0.95)]} max={all_sizes[-1]} words"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
