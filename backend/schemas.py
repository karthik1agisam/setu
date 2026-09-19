"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    scheme: str | None = None
    k: int = Field(default=5, ge=1, le=15)


class EvidenceRef(BaseModel):
    chunk_id: str
    page: int
    source_url: str
    section_path: str = ""


class AskResponse(BaseModel):
    detected_language: str = "en"
    translated_query: str = ""
    transcript: str | None = None
    scheme: str | None = None
    verdict: str
    gate: str
    gate_reason: str = ""
    answer: str | None = None
    missing: list[str] = []
    claim_unsupported_rate: float = 0.0
    assessments: list[dict] = []
    evidence: list[EvidenceRef] = []
    timings_ms: dict[str, float] = {}


class EvidenceResponse(BaseModel):
    chunk_id: str
    scheme: str
    doc: str
    text: str
    section_path: str
    source_url: str
    page_start: int
    page_end: int


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    lang: str = Field(pattern="^(en|hi|te)$")


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    chunks: int
    translator_backend: str
