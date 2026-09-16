"""Tests for RRF fusion + dense retrieval sanity (model-gated)."""

import pytest

from ai.retrieval.hybrid import rrf_fuse


def test_rrf_orders_by_combined_rank():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("c", 5.0), ("a", 4.0), ("d", 3.0)]
    fused = rrf_fuse(dense, bm25)
    ids = [cid for cid, _ in fused]
    # 'a' and 'c' appear in both lists — should outrank single-list items
    assert set(ids[:2]) == {"a", "c"}


def test_rrf_disjoint_lists():
    fused = rrf_fuse([("a", 1.0)], [("b", 1.0)])
    assert {cid for cid, _ in fused} == {"a", "b"}
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_dense_index_sanity():
    """Requires built chunks + model download; skipped when corpus absent."""
    from ai.retrieval.store import ChunkStore

    store = ChunkStore()
    if len(store) == 0:
        pytest.skip("corpus not built")
    from ai.retrieval.dense import DenseIndex

    idx = DenseIndex(store)
    hits = idx.search("income limit for SC scholarship", k=5)
    assert hits
    schemes = {store.get(cid).scheme for cid, _ in hits}
    assert "pmssc" in schemes
