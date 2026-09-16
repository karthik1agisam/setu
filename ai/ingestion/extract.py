"""PDF → per-page line extraction via PyMuPDF.

Stage 1 of ingestion. Keeps line granularity so the cleaner can detect
repeated headers/footers and the structure parser can see clause numbering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


@dataclass
class PageText:
    page: int  # 1-based page number in the source PDF
    lines: list[str] = field(default_factory=list)
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            self.text = "\n".join(self.lines)


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract non-empty stripped lines per page, in reading order."""
    pages: list[PageText] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            raw = page.get_text("text")
            lines = [ln.strip() for ln in raw.splitlines()]
            lines = [ln for ln in lines if ln]
            pages.append(PageText(page=i + 1, lines=lines))
    return pages
