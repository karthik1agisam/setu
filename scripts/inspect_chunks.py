#!/usr/bin/env python3
"""Validate chunk metadata + show samples.

Fails (exit 1) if any chunk is missing provenance fields.

Usage:  uv run python scripts/inspect_chunks.py [--scheme pmkisan] [--samples 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROC = Path("data/processed")
REQUIRED = ["chunk_id", "scheme", "doc", "source_url", "page_start", "text", "n_words"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    files = sorted(PROC.glob("**/*.chunks.jsonl"))
    if args.scheme:
        files = [f for f in files if f.parent.name == args.scheme]
    if not files:
        print("no chunks found — run scripts/chunk_blocks.py first")
        return 1

    bad = 0
    for f in files:
        recs = [json.loads(line) for line in f.open()]
        missing = [
            r["chunk_id"] for r in recs for k in REQUIRED if r.get(k) in (None, "", [])
        ]
        print(f"{f.parent.name}/{f.name}: {len(recs)} chunks, missing_fields={len(missing)}")
        if missing:
            bad += len(missing)
            print(f"  offenders: {missing[:5]}")
        for r in recs[: args.samples]:
            sp = r.get("section_path") or "(preamble)"
            print(f"  {r['chunk_id']} p{r['page_start']}-{r['page_end']} [{sp}]")
            print(f"    {r['text'][:120]}…")

    if bad:
        print(f"\nFAIL: {bad} missing provenance fields")
        return 1
    print("\nOK: all chunks carry full provenance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
