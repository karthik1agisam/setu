#!/usr/bin/env python3
"""TTS loopback eval: synthesize → transcribe → WER vs input text.

A TTS system that ASR can't understand isn't producing intelligible speech —
the loopback WER is a crude but real intelligibility proxy.

Usage:  uv run python evaluation/eval_tts.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ai.speech.asr import ASR
from ai.speech.tts import TTS
from evaluation.eval_asr import cer, wer

OUT = Path("experiments/results/tts_loopback.json")

SAMPLES = {
    "en": "You are eligible for the scholarship.",
    "hi": "आप छात्रवृत्ति के लिए पात्र हैं।",
    "te": "మీరు స్కాలర్‌షిప్ కు అర్హులు.",
}


def main() -> int:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for lang, text in SAMPLES.items():
            wav = TTS.synthesize(text, lang, Path(td) / f"{lang}.wav")
            t = ASR.transcribe(wav, lang=lang)
            w, c = wer(text, t.text), cer(text, t.text)
            rows.append({"lang": lang, "input": text, "asr": t.text,
                         "wer": round(w, 3), "cer": round(c, 3)})
            print(f"{lang}: WER={w:.3f} CER={c:.3f}")
            print(f"   in : {text}")
            print(f"   asr: {t.text}")
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
