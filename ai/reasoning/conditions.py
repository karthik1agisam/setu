"""Schemas for structured eligibility reasoning.

Two layers:
  ConditionAssessment — per-condition status the LLM emits (never a verdict)
  Verdict             — computed by pure code in aggregate.py from statuses
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ConditionKind(StrEnum):
    REQUIRED = "required"  # must hold (e.g. "must be an Indian national")
    EXCLUSION = "exclusion"  # must NOT hold (e.g. "income-tax payers excluded")


class ConditionStatus(StrEnum):
    SATISFIED = "satisfied"  # required cond holds / exclusion not triggered
    VIOLATED = "violated"  # required cond fails / exclusion triggered
    UNKNOWN = "unknown"  # user didn't provide the needed fact


class ConditionAssessment(BaseModel):
    condition_id: str
    description: str
    kind: ConditionKind
    status: ConditionStatus
    evidence_chunk_id: str = Field(description="chunk the condition came from")
    user_fact: str = Field(default="", description="the fact that resolved it, if any")


class ReasoningResult(BaseModel):
    """What the LLM emits — assessments only, no verdict."""

    conditions: list[ConditionAssessment] = Field(default_factory=list)


class VerdictKind(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    CANNOT_DETERMINE = "cannot_determine"


class Verdict(BaseModel):
    verdict: VerdictKind
    conditions_checked: int
    conditions_satisfied: int
    conditions_violated: int
    conditions_unknown: int
    missing: list[str] = Field(default_factory=list)
    assessments: list[ConditionAssessment] = Field(default_factory=list)
