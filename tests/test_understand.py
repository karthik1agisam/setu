"""Unit tests for the Understand stage — schema, JSON extraction, repair.

LLM calls are mocked; Ollama integration is exercised via
scripts/understand_query.py.
"""

from unittest.mock import patch

import pytest

from ai.reasoning.schema import UnderstandingResult
from ai.reasoning.understand import _extract_json, understand

GOOD_JSON = (
    '{"scheme_guess": "pmkisan", "facts": [{"field": "occupation",'
    ' "value": "farmer", "evidence": "I am a farmer"}],'
    ' "missing_fields": ["land_area"], "question_summary": "farmer asks about pmkisan"}'
)


def test_extract_json_clean():
    assert _extract_json(GOOD_JSON)["scheme_guess"] == "pmkisan"


def test_extract_json_with_noise():
    noisy = f"Here is the JSON:\n{GOOD_JSON}\nHope that helps!"
    assert _extract_json(noisy)["scheme_guess"] == "pmkisan"


def test_extract_json_no_object():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


def test_understand_valid_first_try():
    with patch("ai.reasoning.understand.llm.generate", return_value=GOOD_JSON):
        res = understand("I am a farmer with 2 acres")
    assert res.scheme_guess == "pmkisan"
    assert res.facts[0].field == "occupation"


def test_understand_repairs_malformed():
    bad = "not json at all"
    calls = []

    def fake_generate(prompt, **kw):
        calls.append(prompt)
        return bad if len(calls) == 1 else GOOD_JSON

    with patch("ai.reasoning.understand.llm.generate", side_effect=fake_generate):
        res = understand("q")
    assert res.scheme_guess == "pmkisan"
    assert len(calls) == 2  # repair attempt happened


def test_understand_fails_closed_after_max_attempts():
    with patch("ai.reasoning.understand.llm.generate", return_value="garbage"):
        res = understand("q")
    assert res.scheme_guess == "unknown"
    assert res.facts == []


def test_schema_rejects_bad_scheme():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UnderstandingResult.model_validate({"scheme_guess": "nonexistent_scheme"})
