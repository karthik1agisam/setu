"""Cleaning: boilerplate removal, page numbers, hyphenation repair.

Heuristics, applied conservatively:
- A line is boilerplate if it appears in the top/bottom edge of >60% of pages
  (headers/footers repeat; content doesn't).
- Standalone page-number lines are dropped.
- Words hyphenated across line breaks are rejoined.
"""

from __future__ import annotations

import re
from collections import Counter

from ai.ingestion.extract import PageText

BOILERPLATE_MIN_FRAC = 0.6
EDGE_LINES = 3  # how many top/bottom lines count as header/footer candidates

PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?\d{1,4}\s*(of\s*\d{1,4})?\s*$", re.IGNORECASE)
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def find_boilerplate(pages: list[PageText], frac: float = BOILERPLATE_MIN_FRAC) -> set[str]:
    """Lines repeated at page edges across most of the document."""
    n = len(pages)
    if n < 3:
        return set()
    counts: Counter[str] = Counter()
    for p in pages:
        edge = set(p.lines[:EDGE_LINES]) | set(p.lines[-EDGE_LINES:])
        counts.update(edge)
    return {ln for ln, c in counts.items() if c / n >= frac}


def dehyphenate(text: str) -> str:
    """Rejoin 'culti-\\nvable' → 'cultivable'."""
    return HYPHEN_BREAK_RE.sub(r"\1\2", text)


def clean_pages(pages: list[PageText], frac: float = BOILERPLATE_MIN_FRAC) -> list[PageText]:
    boiler = find_boilerplate(pages, frac)
    cleaned: list[PageText] = []
    for p in pages:
        lines = [ln for ln in p.lines if ln not in boiler and not PAGE_NUM_RE.match(ln)]
        text = dehyphenate("\n".join(lines))
        lines = text.splitlines()
        cleaned.append(PageText(page=p.page, lines=lines, text=text))
    return cleaned
