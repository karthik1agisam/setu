#!/usr/bin/env python3
"""Unified multilingual entry: Telugu/Hindi/English → verified answer.

Pipeline: detect → translate-to-English → understand → retrieve → reason →
verdict → conditioned answer → verify → gate → translate answer back.

Usage:  uv run python scripts/ask.py "మీ ప్రశ్న" | "आपका प्रश्न" | "your question" [--scheme pmssc]
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
from ai.translate.detect import detect_language
from ai.translate.indictrans import Translator
from ai.verification.claims import extract_claims
from ai.verification.gate import gate
from ai.verification.verify import verify_claims

ANSWER_PROMPT = """You are SETU. Using ONLY the eligibility assessments below,
write a 2-3 sentence answer. State the verdict plainly, list what was
checked, note missing info. No rules or numbers beyond the assessments.

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

    lang = detect_language(args.question)
    q_en = (
        Translator.translate(args.question, lang, "en") if lang != "en" else args.question
    )

    u = understand(q_en)
    scheme = args.scheme or (u.scheme_guess.value if u.scheme_guess != "unknown" else None)
    if not scheme:
        print(json.dumps({"verdict": "cannot_determine", "reason": "scheme unknown",
                          "detected_language": lang, "translated_query": q_en}))
        return 0

    store = ChunkStore()
    bm25 = BM25Index(store)
    hits = bm25.search(q_en, k=args.k, scheme=scheme)
    evidence = [store.get(cid) for cid, _ in hits]
    ev_texts = [c.text for c in evidence]

    rr = reason(u, evidence)
    verdict = aggregate(rr.conditions)

    amts = "\n".join(
        f"- [{a.status}] {a.description} (evidence: {a.evidence_chunk_id})"
        for a in verdict.assessments
    ) or "(no conditions assessed)"
    gen = generate(
        ANSWER_PROMPT.format(
            verdict=verdict.verdict.value, assessments=amts, question=q_en
        ),
        json_mode=False,
    ).strip()

    cl = extract_claims(gen)
    vr = verify_claims(cl.claims, evidence)
    g = gate(verdict, vr, ev_texts)

    answer_en = gen if g.decision == "release" else None
    answer_out = (
        Translator.translate(answer_en, "en", lang)
        if answer_en and lang != "en"
        else answer_en
    )

    out = {
        "detected_language": lang,
        "translated_query": q_en,
        "scheme": scheme,
        "verdict": verdict.verdict.value,
        "gate": g.decision.value,
        "answer": answer_out,
        "claim_unsupported_rate": vr.unsupported_rate,
        "evidence": [{"chunk_id": c.chunk_id, "page": c.page_start} for c in evidence],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
