"""Chunk store: loads all *.chunks.jsonl into an in-memory corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROC = Path("data/processed")


@dataclass
class ChunkRec:
    chunk_id: str
    scheme: str
    text: str
    section_path: str
    source_url: str
    page_start: int
    page_end: int


class ChunkStore:
    def __init__(self, proc_dir: Path = PROC) -> None:
        self.chunks: list[ChunkRec] = []
        self._by_id: dict[str, ChunkRec] = {}
        for f in sorted(proc_dir.glob("**/*.chunks.jsonl")):
            for line in f.open():
                r = json.loads(line)
                c = ChunkRec(
                    chunk_id=r["chunk_id"],
                    scheme=r["scheme"],
                    text=r["text"],
                    section_path=r.get("section_path", ""),
                    source_url=r["source_url"],
                    page_start=r["page_start"],
                    page_end=r["page_end"],
                )
                self.chunks.append(c)
                self._by_id[c.chunk_id] = c

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> ChunkRec:
        return self._by_id[chunk_id]

    def texts(self, scheme: str | None = None) -> list[str]:
        return [c.text for c in self._filter(scheme)]

    def _filter(self, scheme: str | None) -> list[ChunkRec]:
        return [c for c in self.chunks if scheme is None or c.scheme == scheme]
