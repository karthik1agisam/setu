#!/usr/bin/env python3
"""Synthesize text to speech (te/hi/en).

Usage:  uv run python scripts/speak.py "మీరు అర్హులు" --lang te -o out.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ai.speech.tts import TTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--lang", required=True, choices=["en", "hi", "te"])
    ap.add_argument("-o", "--out", default="data/voice/out.wav")
    args = ap.parse_args()

    t0 = time.time()
    path = TTS.synthesize(args.text, args.lang, Path(args.out))
    print(f"{args.lang} → {path} ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
