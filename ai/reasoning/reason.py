"""Reasoning stage: extracted facts + retrieved evidence → condition assessments.

The LLM reads the retrieved evidence chunks, lists the eligibility conditions
they state (marking each required/exclusion), and assesses each against the
user's facts. It outputs ONLY per-condition assessments — the verdict is
computed deterministically by aggregate.py.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ai.reasoning import llm
from ai.reasoning.conditions import (
    ConditionAssessment,
    ConditionStatus,
    ReasoningResult,
)
from ai.reasoning.schema import UnderstandingResult
from ai.retrieval.store import ChunkRec

PROMPT = """You are the condition-assessment component of SETU, an Indian
welfare-scheme eligibility assistant.

You are given:
1. USER FACTS extracted from the question
2. EVIDENCE CHUNKS from official scheme guidelines (each has an id)

Task:
- From the evidence, list every eligibility condition relevant to the question.
- kind = "required" (must hold) or "exclusion" (must NOT hold).
- For each condition, set status:
    "satisfied" = required condition holds, or exclusion is NOT triggered
    "violated"  = required condition fails, or exclusion IS triggered
    "unknown"   = the user did not provide the fact needed to decide
- evidence_chunk_id must be the id of the chunk the condition comes from.
- user_fact = the user fact that resolved the status, or "".
- Assess ONLY conditions the evidence actually states. Do not invent rules.

Output ONLY JSON:
{{"conditions": [{{"condition_id": "c1", "description": "...", "kind": "required|exclusion",
 "status": "satisfied|violated|unknown", "evidence_chunk_id": "...",
 "user_fact": "..."}}]}}

USER FACTS:
{facts}

EVIDENCE CHUNKS:
{evidence}
"""

REPAIR_PROMPT = """Your previous output was invalid. Error: {error}
Output: {bad}
Return ONLY corrected JSON matching {{"conditions": [...]}}.
"""

MAX_ATTEMPTS = 2


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not m:
        raise ValueError("no JSON object in output")
    return json.loads(m.group())


def _format_facts(u: UnderstandingResult) -> str:
    if not u.facts:
        return "(none stated)"
    return "\n".join(f"- {f.field}: {f.value} (user said: “{f.evidence}”)" for f in u.facts)


def _format_evidence(chunks: list[ChunkRec]) -> str:
    return "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)


def _normalize_chunk_id(raw: str) -> str:
    """Strip brackets/whitespace the LLM adds around chunk ids."""
    return re.sub(r"[\[\]\s]", "", raw)


def _validate_assessments(
    result: ReasoningResult, evidence: list[ChunkRec]
) -> ReasoningResult:
    """Any assessment citing a chunk that was NOT retrieved → UNKNOWN.

    The LLM sometimes emits polluted ids like '[pmkisan:faq:0002] 3' or
    invents ids outright ('pmkisan:faq:00011'). A condition whose evidence
    cannot be pointed at is untrusted, so it can only abstain — this is the
    trust boundary between LLM assessment and deterministic verdict.
    """
    valid = {c.chunk_id for c in evidence}
    out: list[ConditionAssessment] = []
    for a in result.conditions:
        cid = _normalize_chunk_id(a.evidence_chunk_id)
        a = a.model_copy(update={"evidence_chunk_id": cid})
        if cid not in valid:
            a = a.model_copy(update={"status": ConditionStatus.UNKNOWN})
        out.append(a)
    return result.model_copy(update={"conditions": out})


def reason(
    understanding: UnderstandingResult, evidence: list[ChunkRec]
) -> ReasoningResult:
    facts = _format_facts(understanding)
    ev = _format_evidence(evidence)
    last_err: Exception | None = None
    last_out = ""
    for attempt in range(MAX_ATTEMPTS):
        prompt = (
            PROMPT.format(facts=facts, evidence=ev)
            if attempt == 0
            else REPAIR_PROMPT.format(error=last_err, bad=last_out[:800])
        )
        last_out = llm.generate(prompt)
        try:
            res = ReasoningResult.model_validate(_extract_json(last_out))
            return _validate_assessments(res, evidence)
        except (ValueError, ValidationError) as e:
            last_err = e
    return ReasoningResult(conditions=[])  # fail closed → cannot_determine
