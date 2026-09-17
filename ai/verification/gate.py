"""Groundedness gate — the release decision for SETU answers.

Inputs: deterministic verdict + claim-level verification + numeric checks.
Output: release the answer, ask a clarifying question, or abstain.

Rules (fail closed):
  cannot_determine            → clarify (return missing fields)
  verdict + all claims clean  → release
  any claim contradicted      → abstain (evidence contradicts the answer)
  unsupported claims only     → release with caveat + strip risk? no:
                                abstain if >50% unsupported, else release
                                with the unsupported claims listed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ai.reasoning.conditions import Verdict, VerdictKind
from ai.verification.numeric import check_claim_numbers
from ai.verification.verify import VerifyResult


class GateDecision(StrEnum):
    RELEASE = "release"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass
class GateResult:
    decision: GateDecision
    reason: str
    flagged_claims: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def gate(
    verdict: Verdict,
    verify: VerifyResult,
    evidence_texts: list[str],
) -> GateResult:
    if verdict.verdict == VerdictKind.CANNOT_DETERMINE:
        return GateResult(
            GateDecision.CLARIFY,
            "missing information needed to decide",
            missing=verdict.missing,
        )

    # deterministic numeric check on every claim — NLI can't be trusted on numbers
    numeric_flags = []
    for c in verify.claims:
        chk = check_claim_numbers(c.claim, evidence_texts)
        if chk.verdict == "mismatch":
            numeric_flags.append(c.claim)

    if verify.n_contradicted > 0:
        return GateResult(
            GateDecision.ABSTAIN,
            f"{verify.n_contradicted} claim(s) contradict official evidence",
            flagged_claims=[c.claim for c in verify.claims if c.status == "contradicted"],
        )
    if numeric_flags:
        return GateResult(
            GateDecision.ABSTAIN,
            f"{len(numeric_flags)} claim(s) state numbers not present in evidence",
            flagged_claims=numeric_flags,
        )
    if verify.unsupported_rate > 0.5:
        return GateResult(
            GateDecision.ABSTAIN,
            f"unsupported-claim rate {verify.unsupported_rate:.2f} > 0.5",
            flagged_claims=[c.claim for c in verify.claims if c.status == "unsupported"],
        )
    flagged = [c.claim for c in verify.claims if c.status != "supported"]
    return GateResult(
        GateDecision.RELEASE,
        "claims verified" if not flagged else "minor unsupported claims",
        flagged_claims=flagged,
    )
