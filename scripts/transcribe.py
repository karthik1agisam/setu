#!/usr/bin/env python3
"""Transcribe an audio file (te/hi/en).

Usage:  uv run python scripts/transcribe.py path/to/audio.wav [--lang te]
"""

from __future__ import annotations

import argparse
import sys

from ai.speech.asr import ASR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--lang", default=None, choices=["en", "hi", "te"])
    args = ap.parse_args()

    t = ASR.transcribe(args.audio, lang=args.lang)
    print(f"lang={t.language} (p={t.language_prob:.2f}) dur={t.duration_s:.1f}s")
    print(t.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
