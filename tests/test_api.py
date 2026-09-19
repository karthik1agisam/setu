"""API tests — heavy deps (Ollama, NLI, TTS) mocked; schema + routing real."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)

CANNED = {
    "detected_language": "en", "translated_query": "q", "scheme": "pmkisan",
    "verdict": "eligible", "gate": "release", "gate_reason": "claims verified",
    "answer": "You are eligible.", "missing": [], "claim_unsupported_rate": 0.0,
    "assessments": [], "evidence": [], "timings_ms": {"total_ms": 10.0},
}


def test_health():
    with patch("backend.app.ollama_up", return_value=True), \
         patch("backend.app.Translator") as t:
        t.backend.return_value = "nllb"
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks"] >= 0


def test_ask_validates_question_length():
    r = client.post("/ask", json={"question": "ab"})
    assert r.status_code == 422


def test_ask_returns_verdict():
    with patch("backend.app.ollama_up", return_value=True), \
         patch("backend.app.run_pipeline", return_value=CANNED):
        r = client.post("/ask", json={"question": "am i eligible for pmkisan"})
    assert r.status_code == 200
    assert r.json()["verdict"] == "eligible"
    assert r.json()["answer"] == "You are eligible."


def test_ask_503_when_ollama_down():
    with patch("backend.app.ollama_up", return_value=False):
        r = client.post("/ask", json={"question": "am i eligible"})
    assert r.status_code == 503


def test_evidence_404():
    r = client.get("/evidence/nonexistent:chunk:id")
    assert r.status_code == 404


def test_ask_audio_rejects_bad_mime():
    with patch("backend.app.ollama_up", return_value=True):
        r = client.post(
            "/ask/audio",
            files={"file": ("x.txt", b"hello", "text/plain")},
        )
    assert r.status_code == 415
