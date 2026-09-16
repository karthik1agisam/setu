"""Baseline RAG — the unverified arm (B3 in the ablation study).

Retrieves chunks, stuffs them into a prompt, generates a free-text answer.
NO structured conditions, NO verification, NO abstention gate — this is what
a naive "chatbot over documents" does, and it exists so we can measure how
often it fabricates eligibility rules vs. the verified pipeline.
"""

from __future__ import annotations

from ai.reasoning import llm
from ai.retrieval.store import ChunkRec

PROMPT = """You are SETU, an assistant answering questions about Indian welfare
schemes using the official guideline excerpts below.

Answer the user's question directly in 2-4 sentences. Cite chunk ids like
[chunk_id] for the claims you make. If the evidence does not answer the
question, say so plainly.

EVIDENCE:
{evidence}

QUESTION: {question}

ANSWER:"""


def _format_evidence(chunks: list[ChunkRec]) -> str:
    return "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)


def answer(question: str, evidence: list[ChunkRec]) -> str:
    prompt = PROMPT.format(evidence=_format_evidence(evidence), question=question)
    return llm.generate(prompt, json_mode=False).strip()
