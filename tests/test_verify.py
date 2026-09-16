"""Tests for claim extraction parsing + verify aggregation (mocked NLI)."""

from unittest.mock import patch

from ai.verification.claims import Claim, extract_claims
from ai.verification.verify import verify_claims

GOOD = '{"claims": [{"text": "PM-KISAN pays 6000/yr", "cited_chunk_id": "[x:1]"}]}'


def test_extract_claims_parses():
    with patch("ai.verification.claims.llm.generate", return_value=GOOD):
        cl = extract_claims("ans")
    assert len(cl.claims) == 1
    assert cl.claims[0].text == "PM-KISAN pays 6000/yr"


def test_extract_claims_fails_closed():
    with patch("ai.verification.claims.llm.generate", return_value="garbage"):
        cl = extract_claims("ans")
    assert cl.claims == []


def _fake_score(premise, hypothesis):
    # deterministic fake: entail if premise contains a keyword of the claim
    if "6000" in hypothesis and "6000" in premise:
        return {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}
    if "free" in hypothesis and "free" in premise:
        return {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}
    return {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}


def test_verify_aggregation():
    from ai.retrieval.store import ChunkRec

    ev = [ChunkRec("x:1", "s", "benefit of Rs.6000 per annum", "", "u", 1, 1)]
    claims = [
        Claim(text="PM-KISAN pays 6000 per year"),
        Claim(text="eligibility requires voter id"),  # not in evidence
    ]
    with patch("ai.verification.verify.NLIVerifier.score", side_effect=_fake_score):
        vr = verify_claims(claims, ev)
    assert vr.n_supported == 1
    assert vr.n_unsupported == 1
    assert vr.unsupported_rate == 0.5
