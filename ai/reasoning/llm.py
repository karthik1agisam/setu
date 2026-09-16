"""Ollama client — local LLM inference via HTTP.

Default model: qwen2.5:7b-instruct (Apache 2.0). Server must be running
(`ollama serve` or the desktop app). Override via SETU_LLM_MODEL env var.
"""

from __future__ import annotations

import os

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("SETU_LLM_MODEL", "qwen2.5:7b-instruct")


class LLMUnavailableError(RuntimeError):
    pass


def generate(prompt: str, *, json_mode: bool = True, model: str = MODEL) -> str:
    """Single-shot generation. json_mode asks Ollama for JSON-only output."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 512},
    }
    if json_mode:
        payload["format"] = "json"
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180.0)
        r.raise_for_status()
    except httpx.ConnectError as e:
        raise LLMUnavailableError(
            f"Ollama not reachable at {OLLAMA_URL} — run `ollama serve` first"
        ) from e
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise LLMUnavailableError(
                f"model '{model}' not pulled — run `ollama pull {model}`"
            ) from e
        raise
    return r.json()["response"]


def is_available() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False
