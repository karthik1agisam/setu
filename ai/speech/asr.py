"""ASR: faster-whisper large-v3-turbo (CTranslate2, INT8 on CPU).

Transcribes English/Hindi/Telugu audio to text. Audio is decoded via PyAV
(bundled ffmpeg) — wav/aiff/mp3/m4a all work; no system ffmpeg needed.
Audio is processed in memory and never persisted by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_ID = os.environ.get("SETU_ASR_MODEL", "large-v3-turbo")

WHISPER_LANG = {"en": "en", "hi": "hi", "te": "te"}


@dataclass
class Transcript:
    text: str
    language: str  # whisper-detected language code
    language_prob: float
    duration_s: float


class ASR:
    _model: WhisperModel | None = None

    @classmethod
    def get(cls) -> WhisperModel:
        if cls._model is None:
            cls._model = WhisperModel(
                MODEL_ID, device="cpu", compute_type="int8"
            )
        return cls._model

    @classmethod
    def transcribe(cls, audio_path: str | Path, lang: str | None = None) -> Transcript:
        segments, info = cls.get().transcribe(
            str(audio_path),
            language=WHISPER_LANG.get(lang, lang),
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(
            text=text,
            language=info.language,
            language_prob=info.language_probability,
            duration_s=info.duration,
        )
