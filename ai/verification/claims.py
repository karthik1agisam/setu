"""Claim extraction: free-text answer → atomic checkable claims (LLM).

Each claim is a single verifiable statement (one fact, one number, one rule).
The extractor also keeps any [chunk_id] citation the answer made.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError

from ai.reasoning import llm


class Claim(BaseModel):
    text: str = Field(description="one atomic checkable statement")
    cited_chunk_id: str = Field(default="", description="chunk the answer cited, if any")


class ClaimList(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


PROMPT = """Extract the atomic checkable claims from this answer about an Indian
welfare scheme. Rules:
- One claim per JSON item: a single fact, number, or rule.
- Keep any [chunk_id] citation from the answer in cited_chunk_id (strip brackets).
- Do not add claims the answer didn't make. Do not judge truth — just extract.

ANSWER: {answer}

Output ONLY JSON: {{"claims": [{{"text": "...", "cited_chunk_id": "..."}}]}}
"""

REPAIR = """Invalid JSON. Error: {error}
Output: {bad}
Return ONLY {{"claims": [...]}}"""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        raise ValueError("no JSON object")
    return json.loads(m.group())


def extract_claims(answer_text: str) -> ClaimList:
    last_err: Exception | None = None
    last_out = ""
    for attempt in range(2):
        prompt = PROMPT.format(answer=answer_text) if attempt == 0 else REPAIR.format(
            error=last_err, bad=last_out[:800]
        )
        last_out = llm.generate(prompt)
        try:
            return ClaimList.model_validate(_extract_json(last_out))
        except (ValueError, ValidationError) as e:
            last_err = e
    return ClaimList(claims=[])
