"""Tests for numeric checking + the groundedness gate."""

from ai.reasoning.conditions import (
    ConditionAssessment,
    ConditionKind,
    ConditionStatus,
    Verdict,
    VerdictKind,
)
from ai.verification.gate import GateDecision, gate
from ai.verification.numeric import check_claim_numbers, extract_numbers
from ai.verification.verify import ClaimVerdict, VerifyResult


def test_extract_numbers_indian_formats():
    assert 250000.0 in extract_numbers("income is Rs. 2,50,000")
    assert 250000.0 in extract_numbers("income is 2.5 lakh")
    assert 6000.0 in extract_numbers("pays Rs.6000 per annum")
    assert extract_numbers("no numbers here") == []


def test_numeric_mismatch_detected():
    ev = ["income does not exceed Rs. 2,50,000"]
    chk = check_claim_numbers("the limit is Rs. 5 lakh", ev)
    assert chk.verdict == "mismatch"
    chk2 = check_claim_numbers("the limit is 2.5 lakh", ev)
    assert chk2.verdict == "consistent"


def _verdict(kind=VerdictKind.ELIGIBLE, missing=None):
    a = ConditionAssessment(
        condition_id="c1", description="d", kind=ConditionKind.REQUIRED,
        status=ConditionStatus.SATISFIED, evidence_chunk_id="x:1",
    )
    return Verdict(verdict=kind, conditions_checked=1, conditions_satisfied=1,
                   conditions_violated=0, conditions_unknown=0,
                   missing=missing or [], assessments=[a])


def _vr(status="supported", n=1):
    c = ClaimVerdict("claim text", status, "x:1", 0.9, 0.05)
    vr = VerifyResult(claims=[c] * n)
    vr.n_supported = n if status == "supported" else 0
    vr.n_unsupported = n if status == "unsupported" else 0
    vr.n_contradicted = n if status == "contradicted" else 0
    vr.unsupported_rate = 0.0 if status == "supported" else 1.0
    return vr


def test_gate_releases_clean():
    g = gate(_verdict(), _vr("supported"), ["evidence with 6000"])
    assert g.decision == GateDecision.RELEASE


def test_gate_clarifies_unknown():
    v = _verdict(VerdictKind.CANNOT_DETERMINE, missing=["income"])
    g = gate(v, _vr(), ["evidence"])
    assert g.decision == GateDecision.CLARIFY
    assert "income" in g.missing


def test_gate_abstains_on_contradiction():
    g = gate(_verdict(), _vr("contradicted"), ["evidence"])
    assert g.decision == GateDecision.ABSTAIN


def test_gate_abstains_on_numeric_mismatch():
    vr = _vr("supported")
    vr.claims[0].claim = "the limit is 5 lakh"  # evidence says 2,50,000
    g = gate(_verdict(), vr, ["income does not exceed Rs. 2,50,000"])
    assert g.decision == GateDecision.ABSTAIN
    assert "5 lakh" in g.flagged_claims[0]
