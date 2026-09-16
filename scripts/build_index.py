#!/usr/bin/env python3
"""Build + persist the dense FAISS index over all chunks.

Downloads BAAI/bge-m3 on first run (~2.3GB).

Usage:  uv run python scripts/build_index.py
"""

from __future__ import annotations

import sys
import time

from ai.retrieval.dense import DenseIndex
from ai.retrieval.store import ChunkStore


def main() -> int:
    store = ChunkStore()
    if len(store) == 0:
        print("no chunks — run scripts/ingest.py && scripts/chunk_blocks.py first")
        return 1

    t0 = time.time()
    idx = DenseIndex(store)
    path = idx.save()
    print(f"embedded {idx.index.ntotal} chunks in {time.time()-t0:.1f}s → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
