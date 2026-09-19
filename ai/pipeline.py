"""End-to-end verified pipeline — shared by scripts and the backend API.

run_pipeline(question) → structured response dict with verdict, answer,
gate decision, evidence, and per-stage timings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ai.reasoning.aggregate import aggregate
from ai.reasoning.llm import generate
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


@dataclass
class PipelineResult:
    out: dict
    timings_ms: dict[str, float] = field(default_factory=dict)


def run_pipeline(question: str, scheme: str | None = None, k: int = 5,
                 store: ChunkStore | None = None) -> dict:
    t = {}
    t0 = time.perf_counter()

    lang = detect_language(question)
    q_en = Translator.translate(question, lang, "en") if lang != "en" else question
    t["translate_in_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    u = understand(q_en)
    t["understand_ms"] = (time.perf_counter() - t1) * 1000

    scheme = scheme or (u.scheme_guess.value if u.scheme_guess != "unknown" else None)
    if not scheme:
        return {
            "detected_language": lang, "translated_query": q_en,
            "verdict": "cannot_determine", "gate": "clarify",
            "answer": None, "reason": "scheme unknown", "timings_ms": t,
        }

    store = store or ChunkStore()
    bm25 = BM25Index(store)
    hits = bm25.search(q_en, k=k, scheme=scheme)
    evidence = [store.get(cid) for cid, _ in hits]
    ev_texts = [c.text for c in evidence]
    t["retrieve_ms"] = (time.perf_counter() - t1) * 1000 - t["understand_ms"]

    t2 = time.perf_counter()
    rr = reason(u, evidence)
    verdict = aggregate(rr.conditions)
    t["reason_ms"] = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    amts = "\n".join(
        f"- [{a.status}] {a.description} (evidence: {a.evidence_chunk_id})"
        for a in verdict.assessments
    ) or "(no conditions assessed)"
    gen = generate(
        ANSWER_PROMPT.format(verdict=verdict.verdict.value,
                             assessments=amts, question=q_en),
        json_mode=False,
    ).strip()
    cl = extract_claims(gen)
    vr = verify_claims(cl.claims, evidence)
    g = gate(verdict, vr, ev_texts)
    t["verify_gate_ms"] = (time.perf_counter() - t3) * 1000

    answer_en = gen if g.decision == "release" else None
    t4 = time.perf_counter()
    answer_out = (
        Translator.translate(answer_en, "en", lang)
        if answer_en and lang != "en" else answer_en
    )
    t["translate_out_ms"] = (time.perf_counter() - t4) * 1000
    t["total_ms"] = (time.perf_counter() - t0) * 1000

    return {
        "detected_language": lang,
        "translated_query": q_en,
        "scheme": scheme,
        "verdict": verdict.verdict.value,
        "gate": g.decision.value,
        "gate_reason": g.reason,
        "answer": answer_out,
        "missing": g.missing,
        "claim_unsupported_rate": vr.unsupported_rate,
        "assessments": [a.model_dump() for a in verdict.assessments],
        "evidence": [
            {"chunk_id": c.chunk_id, "page": c.page_start,
             "source_url": c.source_url, "section_path": c.section_path}
            for c in evidence
        ],
        "timings_ms": {k: round(v, 1) for k, v in t.items()},
    }
