"""Cross-encoder reranking: BGE-reranker-v2-m3.

Reranks a fused candidate pool (dense + lexical) with a cross-encoder that
scores (query, passage) pairs jointly — far more accurate than bi-encoder
cosine, at the cost of one model pass per candidate. Cheap at our scale
(≤15 candidates per query).
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

MODEL_ID = "BAAI/bge-reranker-v2-m3"


class Reranker:
    _model: CrossEncoder | None = None

    @classmethod
    def get(cls) -> CrossEncoder:
        if cls._model is None:
            cls._model = CrossEncoder(MODEL_ID)
        return cls._model

    @classmethod
    def rerank(
        cls, query: str, candidates: list[tuple[str, float]], store, k: int = 10
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, store.get(cid).text) for cid, _ in candidates]
        scores = cls.get().predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [(candidates[i][0], float(scores[i])) for i in order[:k]]
