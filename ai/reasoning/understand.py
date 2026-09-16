"""Understand stage: user question → structured UnderstandingResult.

The LLM extracts facts; it never judges eligibility. Strict JSON schema with
a bounded repair loop — malformed output gets one corrective retry, then we
return a safe fallback (unknown scheme, no facts) rather than guessing.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ai.reasoning import llm
from ai.reasoning.schema import UnderstandingResult

PROMPT = """You are a fact-extraction component for SETU, an Indian welfare-scheme
eligibility assistant. The schemes you know: pmkisan (farmer income support),
pmjay (health insurance), pmayg (rural housing), pmssc (SC post-matric
scholarship).

Extract what the USER QUESTION states. Rules:
- scheme_guess: best-guess scheme id, or "unknown"
- facts: only facts the user EXPLICITLY stated (income, occupation, land
  ownership, caste category, education, state, family size...). Each fact's
  `evidence` must be a verbatim substring of the question. Never infer.
- missing_fields: eligibility fields the question does not provide that the
  scheme's rules typically need
- question_summary: one-line restatement of what the user is asking

Output ONLY this JSON schema:
{{"scheme_guess": "...", "facts": [{{"field": "...", "value": "...", "evidence": "..."}}],
 "missing_fields": ["..."], "question_summary": "..."}}

USER QUESTION: {question}
"""

REPAIR_PROMPT = """Your previous output was not valid for the required schema.
Error: {error}
Previous output: {bad}

Return ONLY corrected JSON matching:
{{"scheme_guess": "...", "facts": [...], "missing_fields": [...], "question_summary": "..."}}
"""

MAX_ATTEMPTS = 2  # initial + one repair


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of possibly-noisy model output."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in output")
    return json.loads(m.group())


def understand(question: str) -> UnderstandingResult:
    """Extract structured facts; falls back to a safe unknown result."""
    last_err: Exception | None = None
    last_out = ""
    for attempt in range(MAX_ATTEMPTS):
        prompt = (
            PROMPT.format(question=question)
            if attempt == 0
            else REPAIR_PROMPT.format(error=last_err, bad=last_out[:800])
        )
        last_out = llm.generate(prompt)
        try:
            return UnderstandingResult.model_validate(_extract_json(last_out))
        except (ValueError, ValidationError) as e:
            last_err = e
    # fail closed: return a minimal unknown result rather than guessing
    return UnderstandingResult(scheme_guess="unknown", facts=[], missing_fields=[])
