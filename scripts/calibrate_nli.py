#!/usr/bin/env python3
"""Calibrate NLI thresholds on the labeled dev split.

For each (premise, hypothesis, label) in data/benchmark/claims_dev.jsonl we
compute entailment/contradiction probabilities, then sweep thresholds to
maximize label accuracy. Writes experiments/results/nli_calibration.json.

Usage:  uv run python scripts/calibrate_nli.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ai.verification.nli import CONTRA, ENTAIL, NLIVerifier

DEV = Path("data/benchmark/claims_dev.jsonl")
OUT = Path("experiments/results/nli_calibration.json")

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def predict(ent: float, con: float, e_thr: float, c_thr: float) -> str:
    if con >= c_thr:
        return "contradicted"
    if ent >= e_thr:
        return "entailed"
    return "neutral"


def main() -> int:
    pairs = [json.loads(line) for line in DEV.open()]
    print(f"scoring {len(pairs)} labeled pairs…")
    scored = []
    for p in pairs:
        s = NLIVerifier.score(p["premise"], p["hypothesis"])
        scored.append({**p, "entailment": s[ENTAIL], "contradiction": s[CONTRA]})
        print(
            f"  {p['id']}: label={p['label']:<13} ent={s[ENTAIL]:.3f} con={s[CONTRA]:.3f}"
        )

    best = None
    for e_thr in THRESHOLDS:
        for c_thr in THRESHOLDS:
            correct = sum(
                1
                for s in scored
                if predict(s["entailment"], s["contradiction"], e_thr, c_thr) == s["label"]
            )
            acc = correct / len(scored)
            if best is None or acc > best["accuracy"]:
                best = {"entail_thr": e_thr, "contra_thr": c_thr, "accuracy": acc}

    print(
        f"\nbest: entail>={best['entail_thr']} "
        f"contra>={best['contra_thr']} acc={best['accuracy']:.3f}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"best": best, "scored": scored}, indent=2))
    print(f"→ {OUT}")
    print("\nUpdate ENTAIL_THRESHOLD / CONTRA_THRESHOLD in ai/verification/verify.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
