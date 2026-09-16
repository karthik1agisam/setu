"""Claim-level verification: each claim scored against all retrieved chunks.

A claim is SUPPORTED if its best entailment score across retrieved chunks
meets the calibrated threshold; CONTRADICTED if the best contradiction score
meets its threshold; otherwise UNSUPPORTED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.retrieval.store import ChunkRec
from ai.verification.claims import Claim
from ai.verification.nli import CONTRA, ENTAIL, NLIVerifier

# calibrated on the 18-pair labeled dev split (scripts/calibrate_nli.py)
# → experiments/results/nli_calibration.json (accuracy 0.667 — NLI is a weak
# signal on negated/numeric clauses; deterministic checks supplement it)
ENTAIL_THRESHOLD = 0.50
CONTRA_THRESHOLD = 0.50


@dataclass
class ClaimVerdict:
    claim: str
    status: str  # supported | unsupported | contradicted
    best_chunk: str
    entailment: float
    contradiction: float


@dataclass
class VerifyResult:
    claims: list[ClaimVerdict] = field(default_factory=list)
    n_supported: int = 0
    n_unsupported: int = 0
    n_contradicted: int = 0
    unsupported_rate: float = 0.0


def verify_claims(
    claims: list[Claim],
    evidence: list[ChunkRec],
    entail_thr: float = ENTAIL_THRESHOLD,
    contra_thr: float = CONTRA_THRESHOLD,
) -> VerifyResult:
    res = VerifyResult()
    for c in claims:
        best_e, best_c, best_chunk = 0.0, 0.0, ""
        for ch in evidence:
            s = NLIVerifier.score(ch.text, c.text)
            if s[ENTAIL] > best_e:
                best_e, best_chunk = s[ENTAIL], ch.chunk_id
            best_c = max(best_c, s[CONTRA])
        if best_c >= contra_thr:
            status = "contradicted"
            res.n_contradicted += 1
        elif best_e >= entail_thr:
            status = "supported"
            res.n_supported += 1
        else:
            status = "unsupported"
            res.n_unsupported += 1
        res.claims.append(
            ClaimVerdict(c.text, status, best_chunk, round(best_e, 3), round(best_c, 3))
        )
    n = len(res.claims)
    res.unsupported_rate = (res.n_unsupported + res.n_contradicted) / n if n else 0.0
    return res
