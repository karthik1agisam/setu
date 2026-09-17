"""Language detection by Unicode script — no model needed.

Devanagari → Hindi, Telugu block → Telugu, else English. Reliable for our
3-language scope; a mixed-script input is treated as English core text.
"""

from __future__ import annotations

DEVANAGARI = (0x0900, 0x097F)
TELUGU = (0x0C00, 0x0C7F)


def detect_language(text: str) -> str:
    """Return 'hi', 'te', or 'en'."""
    counts = {"hi": 0, "te": 0}
    for ch in text:
        cp = ord(ch)
        if DEVANAGARI[0] <= cp <= DEVANAGARI[1]:
            counts["hi"] += 1
        elif TELUGU[0] <= cp <= TELUGU[1]:
            counts["te"] += 1
    if counts["hi"] == 0 and counts["te"] == 0:
        return "en"
    return "hi" if counts["hi"] >= counts["te"] else "te"
