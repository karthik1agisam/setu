"""Tests for the baseline RAG arm (mocked LLM)."""

from unittest.mock import patch

from ai.reasoning.rag import _format_evidence, answer
from ai.retrieval.store import ChunkRec


def _chunk(cid: str = "s:d:1") -> ChunkRec:
    return ChunkRec(cid, "pmkisan", "Some guideline text.", "1", "https://x", 1, 1)


def test_format_evidence_includes_ids():
    out = _format_evidence([_chunk("a:1"), _chunk("b:2")])
    assert "[a:1]" in out and "[b:2]" in out


def test_answer_calls_llm_and_strips():
    with patch("ai.reasoning.rag.llm.generate", return_value="  yes, eligible \n") as g:
        out = answer("q?", [_chunk()])
    assert out == "yes, eligible"
    assert g.called
