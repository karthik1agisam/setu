#!/usr/bin/env python3
"""Ingest raw PDFs → structured clause blocks with provenance.

Reads data/raw/manifest.csv, runs extract → clean → structure for every
document, writes data/processed/{scheme}/{stem}.blocks.jsonl.

Usage:  uv run python scripts/ingest.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from ai.ingestion.clean import clean_pages
from ai.ingestion.extract import extract_pages
from ai.ingestion.structure import parse_blocks

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
MANIFEST = RAW_DIR / "manifest.csv"


def main() -> int:
    if not MANIFEST.exists():
        print("manifest.csv missing — run scripts/download_docs.py first")
        return 1

    total_blocks = 0
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            if row["machine_readable"] != "True":
                print(f"[skip] {row['scheme']}/{row['filename']} — not machine-readable")
                continue

            pdf = RAW_DIR / row["scheme"] / row["filename"]
            pages = extract_pages(pdf)
            cleaned = clean_pages(pages)
            blocks = parse_blocks(cleaned)

            out = OUT_DIR / row["scheme"] / f"{Path(row['filename']).stem}.blocks.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as fh:
                for i, b in enumerate(blocks):
                    rec = {
                        "block_id": f"{row['scheme']}:{Path(row['filename']).stem}:{i}",
                        "scheme": row["scheme"],
                        "doc": row["filename"],
                        "doc_version": row["doc_version"],
                        "source_url": row["source_url"],
                        "page": b.page,
                        "section_path": b.section_path,
                        "text": b.text,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            structured = sum(1 for b in blocks if b.section_path)
            print(
                f"[{row['scheme']}] {row['filename']}: "
                f"{len(pages)} pages → {len(blocks)} blocks "
                f"({structured} structured)"
            )
            total_blocks += len(blocks)

    print(f"\nTotal blocks: {total_blocks} → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
