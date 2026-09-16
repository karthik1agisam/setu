"""Dense retrieval: BGE-M3 embeddings → FAISS flat inner-product index.

BGE-M3 (BAAI/bge-m3): 100+ languages, 1024-dim dense vectors, MIT license.
Embeddings are L2-normalized so inner product == cosine similarity.
Index persists to data/processed/index/ (gitignored, rebuilt by
scripts/build_index.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ai.retrieval.store import ChunkStore

INDEX_DIR = Path("data/processed/index")
MODEL_ID = "BAAI/bge-m3"


class Embedder:
    """Lazy-loaded BGE-M3 embedder (heavy model — only loaded on use)."""

    _model: SentenceTransformer | None = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._model is None:
            cls._model = SentenceTransformer(MODEL_ID)
        return cls._model

    @classmethod
    def encode(cls, texts: list[str]) -> np.ndarray:
        emb = cls.get().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


class DenseIndex:
    def __init__(self, store: ChunkStore, scheme: str | None = None) -> None:
        self.store = store
        self.recs = store._filter(scheme)
        self.ids = [c.chunk_id for c in self.recs]
        self.index = faiss.IndexFlatIP(1024)
        if self.recs:
            self.index.add(Embedder.encode([c.text for c in self.recs]))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        q = Embedder.encode([query])
        scores, idxs = self.index.search(q, min(k, self.index.ntotal))
        return [
            (self.ids[i], float(s))
            for s, i in zip(scores[0], idxs[0], strict=False)
            if 0 <= i < len(self.ids)
        ]

    def save(self, scheme: str | None = None) -> Path:
        name = scheme or "all"
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_DIR / f"dense_{name}.faiss"))
        (INDEX_DIR / f"dense_{name}.ids.json").write_text(json.dumps(self.ids))
        return INDEX_DIR / f"dense_{name}.faiss"
