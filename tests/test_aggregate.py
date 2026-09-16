"""Truth-table tests for the deterministic verdict aggregator.

The aggregator is the trust boundary — it must be exactly correct.
"""

import pytest

from ai.reasoning.aggregate import aggregate
from ai.reasoning.conditions import (
    ConditionAssessment,
    ConditionKind,
    ConditionStatus,
    VerdictKind,
)

R = ConditionKind.REQUIRED
X = ConditionKind.EXCLUSION
S = ConditionStatus.SATISFIED
V = ConditionStatus.VIOLATED
U = ConditionStatus.UNKNOWN


def _a(kind: ConditionKind, status: ConditionStatus, cid: str = "c1") -> ConditionAssessment:
    return ConditionAssessment(
        condition_id=cid,
        description="d",
        kind=kind,
        status=status,
        evidence_chunk_id="x:0",
    )


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([(R, S), (R, S), (X, S)], VerdictKind.ELIGIBLE),
        ([(R, S), (X, V)], VerdictKind.INELIGIBLE),  # exclusion triggered
        ([(R, V)], VerdictKind.INELIGIBLE),  # required fails
        ([(R, S), (R, U)], VerdictKind.CANNOT_DETERMINE),  # missing fact
        ([(X, U)], VerdictKind.CANNOT_DETERMINE),  # can't rule out exclusion
        ([], VerdictKind.CANNOT_DETERMINE),  # no evidence → abstain
        ([(R, S), (X, V), (R, U)], VerdictKind.INELIGIBLE),  # violation beats unknown
    ],
)
def test_truth_table(statuses, expected):
    assessments = [_a(k, s, f"c{i}") for i, (k, s) in enumerate(statuses)]
    assert aggregate(assessments).verdict == expected


def test_missing_lists_unknown_conditions():
    v = aggregate([_a(R, S, "c0"), _a(R, U, "c1")])
    assert v.verdict == VerdictKind.CANNOT_DETERMINE
    assert any("c1" in m for m in v.missing)


def test_counts():
    v = aggregate([_a(R, S), _a(R, V, "c2"), _a(X, S, "c3")])
    assert (v.conditions_checked, v.conditions_satisfied, v.conditions_violated) == (3, 2, 1)


def test_unretrieved_evidence_forces_unknown():
    """Trust boundary: assessment citing a chunk that wasn't retrieved → UNKNOWN."""
    from ai.reasoning.conditions import ReasoningResult
    from ai.reasoning.reason import _validate_assessments
    from ai.retrieval.store import ChunkRec

    good = _a(R, S, "c1")
    bad = _a(R, S, "c2")
    bad = bad.model_copy(update={"evidence_chunk_id": "[invented:999] 3"})
    res = ReasoningResult(conditions=[good, bad])
    evidence = [ChunkRec("real:1", "s", "t", "", "u", 1, 1)]
    good = good.model_copy(update={"evidence_chunk_id": "real:1"})
    res = ReasoningResult(conditions=[good, bad])
    out = _validate_assessments(res, evidence)
    assert out.conditions[0].status == S
    assert out.conditions[1].status == U  # hallucinated citation → unknown
