"""NLI verification: mDeBERTa XNLI scores (premise=evidence, hypothesis=claim).

Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
Labels: entailment / neutral / contradiction. This is a *signal*, not proof —
thresholds are calibrated on a labeled dev split (scripts/calibrate_nli.py).
"""

from __future__ import annotations

from transformers import pipeline

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

ENTAIL = "entailment"
NEUTRAL = "neutral"
CONTRA = "contradiction"


class NLIVerifier:
    _pipe = None

    @classmethod
    def get(cls):
        if cls._pipe is None:
            cls._pipe = pipeline(
                "text-classification", model=MODEL_ID, top_k=None
            )
        return cls._pipe

    @classmethod
    def score(cls, premise: str, hypothesis: str) -> dict[str, float]:
        """Return {entailment, neutral, contradiction} probabilities."""
        out = cls.get()(
            {"text": premise, "text_pair": hypothesis}, truncation=True, max_length=512
        )
        scores = {r["label"].lower(): r["score"] for r in out}
        return {
            ENTAIL: scores.get(ENTAIL, 0.0),
            NEUTRAL: scores.get(NEUTRAL, 0.0),
            CONTRA: scores.get(CONTRA, 0.0),
        }
