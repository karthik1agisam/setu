"""Deterministic numeric checking — the complement to weak NLI.

NLI measurably fails on numeric paraphrases and reversed thresholds
(calibration: dev acc 0.667). This module is rule-based and exact: extract
every monetary/quantity number from a claim, normalize Indian formats
("2.5 lakh", "Rs. 2,50,000", "Rs.6000"), and check whether the same value
appears in the evidence. A claim stating a number absent from evidence gets
flagged — pure code, no model, no brittleness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUM_RE = re.compile(
    r"(?:rs\.?\s*|₹\s*|inr\s*)?(\d[\d,]*(?:\.\d+)?)\s*"
    r"(lakh|lakhs|crore|crores|thousand|k\b|/-|per\s+(?:annum|year|month))?",
    re.IGNORECASE,
)

MULTIPLIERS = {
    "lakh": 100_000,
    "lakhs": 100_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "thousand": 1_000,
    "k": 1_000,
}


def _norm(num_str: str, unit: str | None) -> float | None:
    try:
        v = float(num_str.replace(",", ""))
    except ValueError:
        return None
    if unit:
        u = unit.lower().strip()
        for key, mult in MULTIPLIERS.items():
            if u.startswith(key):
                return v * mult
    return v


def extract_numbers(text: str) -> list[float]:
    """All normalized numeric values in text (lakh/crore→absolute)."""
    vals = []
    for m in NUM_RE.finditer(text):
        v = _norm(m.group(1), m.group(2))
        if v is not None:
            vals.append(v)
    return vals


@dataclass
class NumericCheck:
    claim: str
    claim_numbers: list[float]
    evidence_numbers: list[float]
    verdict: str  # "consistent" | "mismatch" | "no_numbers"


def check_claim_numbers(claim: str, evidence_texts: list[str]) -> NumericCheck:
    """If the claim states numbers, every number must appear in evidence.

    '2.5 lakh' in claim matches '2,50,000' or 'Rs. 250000' in evidence.
    A claim number absent from all evidence = mismatch (deterministic
    contradiction signal NLI misses).
    """
    claim_nums = extract_numbers(claim)
    if not claim_nums:
        return NumericCheck(claim, [], [], "no_numbers")
    ev_nums: list[float] = []
    for t in evidence_texts:
        ev_nums += extract_numbers(t)
    ev_set = set(ev_nums)
    missing = [n for n in claim_nums if n not in ev_set]
    return NumericCheck(
        claim, claim_nums, ev_nums, "mismatch" if missing else "consistent"
    )
