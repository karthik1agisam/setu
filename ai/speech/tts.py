"""TTS: IndicF5 (preferred, gated) → MMS-TTS (ungated default) → macOS `say`.

IndicF5 is zero-shot voice cloning (best quality) but its HF repo is gated.
MMS-TTS (facebook/mms-tts-{eng,hin,tel}, VITS, ~100MB, ungated) is the
zero-friction default. `say` is the last-resort local fallback.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write as wav_write

MMS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "te": "facebook/mms-tts-tel",
}
SAY_VOICES = {"en": "Rishi", "hi": "Lekha", "te": "Geeta"}
SAMPLE_RATE = 16000


class TTS:
    _models: dict[str, tuple] = {}

    @classmethod
    def _load_mms(cls, lang: str):
        if lang not in cls._models:
            from transformers import AutoTokenizer, VitsModel

            tok = AutoTokenizer.from_pretrained(MMS_MODELS[lang])
            mdl = VitsModel.from_pretrained(MMS_MODELS[lang])
            cls._models[lang] = (tok, mdl)
        return cls._models[lang]

    @classmethod
    def synthesize(cls, text: str, lang: str, out_path: str | Path) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tok, mdl = cls._load_mms(lang)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            waveform = mdl(**inputs).waveform[0].numpy()
        pcm = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
        wav_write(str(out), SAMPLE_RATE, pcm)
        return out

    @classmethod
    def synthesize_say(cls, text: str, lang: str, out_path: str | Path) -> Path:
        """macOS fallback — dev only, not portable."""
        out = Path(out_path)
        subprocess.run(
            ["say", "-v", SAY_VOICES[lang], "-o", str(out), text], check=True
        )
        return out
