#!/usr/bin/env python3
"""ASR eval: WER + CER over a labeled audio manifest.

Manifest: data/benchmark/asr_eval.jsonl —
  {"audio": "path.wav", "reference": "expected text", "lang": "en"}

Generate English/Hindi/Telugu samples with macOS `say` (see
scripts/make_asr_samples.sh) or record your own voice.

Usage:  uv run python evaluation/eval_asr.py
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from ai.speech.asr import ASR

MANIFEST = Path("data/benchmark/asr_eval.jsonl")
OUT = Path("experiments/results/asr_eval.json")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a: list, b: list) -> int:
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (x != b[j - 1]))
    return dp[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = _normalize(ref).split(), _normalize(hyp).split()
    return _edit_distance(r, h) / max(len(r), 1)


def cer(ref: str, hyp: str) -> float:
    r, h = list(_normalize(ref)), list(_normalize(hyp))
    return _edit_distance(r, h) / max(len(r), 1)


def main() -> int:
    if not MANIFEST.exists():
        print(f"{MANIFEST} missing — create eval audio first (scripts/make_asr_samples.sh)")
        return 1
    items = [json.loads(line) for line in MANIFEST.open()]
    rows = []
    for it in items:
        t = ASR.transcribe(it["audio"], lang=it["lang"])
        w, c = wer(it["reference"], t.text), cer(it["reference"], t.text)
        rows.append({"id": it["audio"], "lang": it["lang"], "ref": it["reference"],
                     "hyp": t.text, "wer": round(w, 3), "cer": round(c, 3)})
        print(f"{it['lang']} {Path(it['audio']).name}: WER={w:.3f} CER={c:.3f}")
        print(f"   ref: {it['reference'][:80]}")
        print(f"   hyp: {t.text[:80]}")

    by_lang: dict[str, list] = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r)
    summary = {
        lang: {
            "n": len(rs),
            "wer": round(sum(r["wer"] for r in rs) / len(rs), 3),
            "cer": round(sum(r["cer"] for r in rs) / len(rs), 3),
        }
        for lang, rs in by_lang.items()
    }
    OUT.write_text(json.dumps({"per_lang": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
