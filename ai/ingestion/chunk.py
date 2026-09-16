"""Clause blocks → retrieval chunks.

Policy (per master plan Part 9):
- one chunk ≈ one clause; target ~150–450 tokens (≈110–340 words at 1.33 w/tok)
- short blocks merge with consecutive siblings sharing a section prefix —
  an exclusion list like (i)..(vi) is meaningless split apart
- oversized blocks split at sentence boundaries, never mid-sentence;
  every split part keeps the full section_path
- provenance (source_url, page range, doc_version) on every chunk
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MIN_WORDS = 110
MAX_WORDS = 340
HARD_MAX_WORDS = 500  # single unsplittable sentences may exceed MAX

SENT_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class Chunk:
    chunk_id: str
    scheme: str
    doc: str
    doc_version: str
    source_url: str
    section_paths: list[str]
    page_start: int
    page_end: int
    n_words: int
    text: str


def _words(text: str) -> int:
    return len(text.split())


def _sentences(text: str) -> list[str]:
    parts = SENT_SPLIT_RE.split(text)
    return [p for p in parts if p.strip()]


def _common_prefix(paths: list[str]) -> str:
    """Deepest common section prefix across merged blocks."""
    if not paths:
        return ""
    parts = [p.split(" > ") for p in paths if p]
    if not parts:
        return ""
    pref: list[str] = []
    for comp in zip(*parts, strict=False):
        if len(set(comp)) == 1:
            pref.append(comp[0])
        else:
            break
    return " > ".join(pref)


def chunk_blocks(records: list[dict], doc_stem: str) -> list[Chunk]:
    """records = parsed lines from a .blocks.jsonl file."""
    chunks: list[Chunk] = []
    # stage 1: merge consecutive blocks until we reach MIN_WORDS,
    # or stop early if adding the next block would exceed MAX_WORDS
    i = 0
    n = len(records)
    seq = 0

    def emit(text: str, rec_group: list[dict]) -> None:
        nonlocal seq
        seq += 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc_stem}:{seq:04d}",
                scheme=rec_group[0]["scheme"],
                doc=rec_group[0]["doc"],
                doc_version=rec_group[0]["doc_version"],
                source_url=rec_group[0]["source_url"],
                section_paths=[r["section_path"] for r in rec_group if r["section_path"]],
                page_start=min(r["page_start"] for r in rec_group),
                page_end=max(r["page_end"] for r in rec_group),
                n_words=_words(text),
                text=text,
            )
        )

    while i < n:
        group = [records[i]]
        text = records[i]["text"]
        w = _words(text)

        if w > MAX_WORDS:
            # stage 2: split oversized clause at sentence boundaries
            sents = _sentences(text)
            cur, cur_w = [], 0
            for s in sents:
                sw = _words(s)
                if cur and cur_w + sw > MAX_WORDS:
                    emit(" ".join(cur), group)
                    cur, cur_w = [], 0
                cur.append(s)
                cur_w += sw
            if cur:
                emit(" ".join(cur), group)
            i += 1
            continue

        # merge forward while under MIN_WORDS and next block fits;
        # never absorb an oversized block — it must go through the split path
        j = i + 1
        while j < n and (w < MIN_WORDS or w + _words(records[j]["text"]) <= MAX_WORDS):
            nxt_w = _words(records[j]["text"])
            if nxt_w > MAX_WORDS:
                break
            if w >= MIN_WORDS and w + nxt_w > MAX_WORDS:
                break
            if w < MIN_WORDS and w + nxt_w > HARD_MAX_WORDS:
                break
            group.append(records[j])
            text = text + " " + records[j]["text"]
            w += nxt_w
            j += 1
        emit(text, group)
        i = j

    return chunks


def load_blocks(blocks_file: Path) -> list[dict]:
    return [json.loads(line) for line in blocks_file.open()]


def chunk_document(blocks_file: Path, out_file: Path) -> list[Chunk]:
    recs = load_blocks(blocks_file)
    stem = blocks_file.stem.removesuffix(".blocks")
    chunks = chunk_blocks(recs, doc_stem=f"{recs[0]['scheme']}:{stem}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as fh:
        for c in chunks:
            fh.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "scheme": c.scheme,
                        "doc": c.doc,
                        "doc_version": c.doc_version,
                        "source_url": c.source_url,
                        "section_path": _common_prefix(c.section_paths),
                        "section_paths": c.section_paths,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "n_words": c.n_words,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return chunks
