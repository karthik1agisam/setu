#!/usr/bin/env python3
"""Full verified pipeline: understand → retrieve → reason → verdict →
conditioned answer → claim extraction → NLI+numeric verify → gate.

Usage:  uv run python scripts/answer_query.py "question" [--scheme pmkisan]
"""

from __future__ import annotations

import argparse
import json
import sys

from ai.reasoning.aggregate import aggregate
from ai.reasoning.llm import generate, is_available
from ai.reasoning.reason import reason
from ai.reasoning.understand import understand
from ai.retrieval.bm25 import BM25Index
from ai.retrieval.store import ChunkStore
from ai.verification.claims import extract_claims
from ai.verification.gate import gate
from ai.verification.verify import verify_claims

ANSWER_PROMPT = """You are SETU. Using ONLY the eligibility assessments below
(each tied to an official evidence chunk), write a 2-3 sentence answer to the
user. State the verdict plainly, list what was checked, and note any missing
information. Do not add rules or numbers not in the assessments.

VERDICT: {verdict}
ASSESSMENTS:
{assessments}

QUESTION: {question}

ANSWER:"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    if not is_available():
        print("Ollama not running")
        return 1

    u = understand(args.question)
    scheme = args.scheme or (u.scheme_guess.value if u.scheme_guess != "unknown" else None)
    if not scheme:
        print(json.dumps({"verdict": "cannot_determine", "reason": "scheme unknown"}))
        return 0

    store = ChunkStore()
    bm25 = BM25Index(store)
    hits = bm25.search(args.question, k=args.k, scheme=scheme)
    evidence = [store.get(cid) for cid, _ in hits]
    ev_texts = [c.text for c in evidence]

    rr = reason(u, evidence)
    verdict = aggregate(rr.conditions)

    # generate a conditioned answer from the verdict + assessments
    amts = "\n".join(
        f"- [{a.status}] {a.description} (evidence: {a.evidence_chunk_id})"
        for a in verdict.assessments
    ) or "(no conditions assessed)"
    gen = generate(
        ANSWER_PROMPT.format(
            verdict=verdict.verdict.value, assessments=amts, question=args.question
        ),
        json_mode=False,
    ).strip()

    cl = extract_claims(gen)
    vr = verify_claims(cl.claims, evidence)
    g = gate(verdict, vr, ev_texts)

    out = {
        "question": args.question,
        "scheme": scheme,
        "verdict": verdict.verdict.value,
        "gate": {"decision": g.decision.value, "reason": g.reason,
                 "flagged_claims": g.flagged_claims, "missing": g.missing},
        "answer": gen if g.decision == "release" else None,
        "assessments": [a.model_dump() for a in verdict.assessments],
        "claim_verification": {
            "n_claims": len(vr.claims),
            "unsupported_rate": vr.unsupported_rate,
        },
        "evidence": [
            {"chunk_id": c.chunk_id, "page": c.page_start, "url": c.source_url}
            for c in evidence
        ],
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
