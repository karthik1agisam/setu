"""Hybrid retrieval: reciprocal-rank fusion of dense + lexical rankings."""

from __future__ import annotations

from ai.retrieval.bm25 import BM25Index
from ai.retrieval.dense import DenseIndex

RRF_K = 60  # standard RRF constant


def rrf_fuse(
    *rankings: list[tuple[str, float]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """RRF score = Σ 1/(k + rank). Rankings are (chunk_id, score) lists."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class HybridIndex:
    def __init__(self, bm25: BM25Index, dense: DenseIndex) -> None:
        self.bm25 = bm25
        self.dense = dense

    def search(
        self,
        query: str,
        k: int = 10,
        scheme: str | None = None,
        k_dense: int = 10,
        k_bm25: int = 10,
    ) -> list[tuple[str, float]]:
        d = self.dense.search(query, k=k_dense)
        b = self.bm25.search(query, k=k_bm25, scheme=scheme)
        return rrf_fuse(d, b)[:k]
