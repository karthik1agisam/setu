"""Schemas for the Understand stage (query → structured facts).

The LLM's job is ONLY to extract what the user said — not to judge
eligibility. Output validates against UnderstandingResult; anything else is
rejected and sent through the repair loop.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Scheme(StrEnum):
    PMKISAN = "pmkisan"
    PMJAY = "pmjay"
    PMAYG = "pmayg"
    PMSSC = "pmssc"
    UNKNOWN = "unknown"


class Fact(BaseModel):
    """A user-stated fact relevant to eligibility (e.g. income, occupation)."""

    field: str = Field(description="snake_case name, e.g. annual_income, occupation")
    value: str = Field(description="the value as stated, e.g. '300000', 'teacher'")
    evidence: str = Field(description="verbatim substring of the query supporting it")


class UnderstandingResult(BaseModel):
    scheme_guess: Scheme
    facts: list[Fact] = Field(default_factory=list)
    missing_fields: list[str] = Field(
        default_factory=list,
        description="fields the scheme's rules need but the user didn't provide",
    )
    question_summary: str = ""
