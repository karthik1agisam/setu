"""SETU FastAPI backend.

Endpoints:
  GET  /health              service + dependency status
  POST /ask                 text query → verified answer
  POST /ask/audio           audio file → ASR → verified answer
  GET  /evidence/{chunk_id} chunk detail for evidence cards
  POST /tts                 text → wav

Uploaded audio is written to a temp file for ASR decoding, deleted
immediately, never persisted.
"""

from __future__ import annotations

import tempfile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ai.pipeline import run_pipeline
from ai.reasoning.llm import is_available as ollama_up
from ai.retrieval.store import ChunkStore
from ai.speech.asr import ASR
from ai.speech.tts import TTS
from ai.translate.indictrans import Translator
from backend.schemas import (
    AskRequest,
    AskResponse,
    EvidenceResponse,
    HealthResponse,
    TTSRequest,
)

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_AUDIO_PREFIXES = ("audio/", "application/octet-stream")

app = FastAPI(title="SETU", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_store: ChunkStore | None = None


def store() -> ChunkStore:
    global _store
    if _store is None:
        _store = ChunkStore()
    return _store


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        ollama=ollama_up(),
        chunks=len(store()),
        translator_backend=Translator.backend(),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not ollama_up():
        raise HTTPException(503, "Ollama not running")
    return AskResponse(**run_pipeline(req.question, req.scheme, req.k, store()))


@app.post("/ask/audio", response_model=AskResponse)
async def ask_audio(file: Annotated[UploadFile, File()]):
    if not ollama_up():
        raise HTTPException(503, "Ollama not running")
    if file.content_type and not file.content_type.startswith(ALLOWED_AUDIO_PREFIXES):
        raise HTTPException(415, f"unsupported content type {file.content_type}")
    data = await file.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio too large (max 10MB)")
    if not data:
        raise HTTPException(400, "empty file")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        transcript = ASR.transcribe(tmp.name)
    out = run_pipeline(transcript.text, store=store())
    out["transcript"] = transcript.text
    out["detected_language"] = transcript.language
    return AskResponse(**out)


@app.get("/evidence/{chunk_id:path}", response_model=EvidenceResponse)
def evidence(chunk_id: str):
    try:
        c = store().get(chunk_id)
    except KeyError:
        raise HTTPException(404, "chunk not found") from None
    # ChunkRec carries text+provenance; doc/version live in the chunks file —
    # expose what the store knows.
    return EvidenceResponse(
        chunk_id=c.chunk_id,
        scheme=c.scheme,
        doc=c.chunk_id.split(":")[1],
        text=c.text,
        section_path=c.section_path,
        source_url=c.source_url,
        page_start=c.page_start,
        page_end=c.page_end,
    )


@app.post("/tts")
def tts(req: TTSRequest):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = TTS.synthesize(req.text, req.lang, tmp.name)
        return FileResponse(path, media_type="audio/wav", filename="setu.wav")
    except Exception as e:  # noqa: BLE001 — surface as 500 with context
        raise HTTPException(500, f"TTS failed: {e}") from e
