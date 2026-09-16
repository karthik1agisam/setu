"""Lexical retrieval baseline: BM25 over chunk text."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ai.retrieval.store import ChunkStore

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 over the chunk corpus. Rebuilt per search only if the scheme
    filter changes — corpus is ~170 chunks, so this is cheap either way."""

    def __init__(self, store: ChunkStore) -> None:
        self.store = store
        self._cache: dict[str | None, tuple[BM25Okapi, list[str]]] = {}

    def _index(self, scheme: str | None) -> tuple[BM25Okapi, list[str]]:
        if scheme not in self._cache:
            recs = self.store._filter(scheme)
            corpus = [tokenize(c.text) for c in recs]
            self._cache[scheme] = (BM25Okapi(corpus), [c.chunk_id for c in recs])
        return self._cache[scheme]

    def search(self, query: str, k: int = 10, scheme: str | None = None) -> list[tuple[str, float]]:
        idx, ids = self._index(scheme)
        scores = idx.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(ids[i], float(scores[i])) for i in top if scores[i] > 0]
