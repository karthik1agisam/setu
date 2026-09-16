#!/usr/bin/env python3
"""Inspect ingestion output: block counts, structure coverage, samples.

Usage:  uv run python scripts/inspect_ingestion.py [--scheme pmkisan]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT_DIR = Path("data/processed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    files = sorted(OUT_DIR.glob("**/*.blocks.jsonl"))
    if args.scheme:
        files = [f for f in files if f.parent.name == args.scheme]
    if not files:
        print("no blocks found — run scripts/ingest.py first")
        return 1

    failures = 0
    for f in files:
        recs = [json.loads(line) for line in f.open()]
        structured = sum(1 for r in recs if r["section_path"])
        missing_prov = sum(1 for r in recs if not r["source_url"] or not r["page_start"])
        pct = 100 * structured / max(len(recs), 1)
        print(f"\n=== {f.parent.name}/{f.name}")
        print(
            f"blocks={len(recs)}  structured={structured} ({pct:.0f}%)  "
            f"missing_provenance={missing_prov}"
        )
        if missing_prov:
            failures += 1
        for r in recs[: args.samples]:
            path = r["section_path"] or "(preamble)"
            print(f"  p{r['page_start']}-{r['page_end']} [{path}] {r['text'][:110]}…")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
