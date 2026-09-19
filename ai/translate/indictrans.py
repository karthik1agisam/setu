"""Indic translation: IndicTrans2 (preferred) with NLLB-200 fallback.

IndicTrans2 (AI4Bharat, MIT) is the plan's primary choice, but its HF repos
are gated — they need an HF account + license acceptance. For a fully
ungated zero-friction path we fall back to facebook/nllb-200-distilled-600M
(NLLB covers eng_Latn / hin_Deva / tel_Telu). Set HF_TOKEN with access to
ai4bharat/indictrans2-* to use the better model.

Core reasoning stays in English — official documents are English.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FLORES = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu"}
INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
NLLB = "facebook/nllb-200-distilled-600M"


class Translator:
    """Lazy model cache; IndicTrans2 if reachable, else NLLB."""

    _cache: dict[str, tuple] = {}
    _backend: str | None = None
    _ip = None

    # -- backend selection -------------------------------------------------
    @classmethod
    def _use_indictrans(cls) -> bool:
        if cls._backend is not None:
            return cls._backend == "indictrans2"
        try:
            from huggingface_hub import hf_hub_download
            from IndicTransToolkit import IndicProcessor

            # metadata is public even on gated repos — probe an actual file
            hf_hub_download(INDIC_EN, "config.json")
            cls._ip = IndicProcessor(inference=True)
            cls._backend = "indictrans2"
        except Exception:
            cls._backend = "nllb"
        return cls._backend == "indictrans2"

    # -- backends ----------------------------------------------------------
    @classmethod
    def _load(cls, model_id: str):
        if model_id not in cls._cache:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.float32
            )
            cls._cache[model_id] = (tok, mdl)
        return cls._cache[model_id]

    @classmethod
    def _it2(cls, text: str, src: str, tgt: str) -> str:
        model_id = INDIC_EN if tgt == "en" else EN_INDIC
        tok, mdl = cls._load(model_id)
        batch = cls._ip.preprocess_batch([text], FLORES[src], FLORES[tgt])
        inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt")
        with torch.no_grad():
            out = mdl.generate(**inputs, max_length=256, num_beams=4)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        return cls._ip.postprocess_batch(decoded, lang=FLORES[tgt])[0]

    @classmethod
    def _nllb(cls, text: str, src: str, tgt: str) -> str:
        tok, mdl = cls._load(NLLB)
        tok.src_lang = FLORES[src]
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                forced_bos_token_id=tok.convert_tokens_to_ids(FLORES[tgt]),
                max_length=256,
            )
        return tok.batch_decode(out, skip_special_tokens=True)[0]

    # -- public ------------------------------------------------------------
    @classmethod
    def translate(cls, text: str, src: str, tgt: str) -> str:
        if src == tgt:
            return text
        return cls._it2(text, src, tgt) if cls._use_indictrans() else cls._nllb(text, src, tgt)

    @classmethod
    def backend(cls) -> str:
        cls._use_indictrans()
        return cls._backend or "unknown"
