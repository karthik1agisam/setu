"""Deterministic verdict aggregation — pure code, fully testable.

Rules (AND over required conditions, OR over exclusions):
  - any exclusion VIOLATED            → ineligible
  - any required VIOLATED             → ineligible
  - any required/exclusion UNKNOWN    → cannot_determine
  - all assessed, none unknown        → eligible
  - no conditions at all              → cannot_determine

The LLM's per-condition status is trusted only as an assessment; the
verdict itself is computed here so a hallucinated boolean can't slip through.
"""

from __future__ import annotations

from ai.reasoning.conditions import (
    ConditionAssessment,
    ConditionStatus,
    Verdict,
    VerdictKind,
)


def aggregate(assessments: list[ConditionAssessment]) -> Verdict:
    violated = [a for a in assessments if a.status == ConditionStatus.VIOLATED]
    unknown = [a for a in assessments if a.status == ConditionStatus.UNKNOWN]
    satisfied = [a for a in assessments if a.status == ConditionStatus.SATISFIED]

    if not assessments:
        verdict = VerdictKind.CANNOT_DETERMINE
        missing = ["no conditions could be extracted from evidence"]
    elif violated:
        verdict = VerdictKind.INELIGIBLE
        missing = []
    elif unknown:
        verdict = VerdictKind.CANNOT_DETERMINE
        missing = [f"{a.condition_id}: {a.description}" for a in unknown]
    else:
        verdict = VerdictKind.ELIGIBLE
        missing = []

    return Verdict(
        verdict=verdict,
        conditions_checked=len(assessments),
        conditions_satisfied=len(satisfied),
        conditions_violated=len(violated),
        conditions_unknown=len(unknown),
        missing=missing,
        assessments=assessments,
    )
