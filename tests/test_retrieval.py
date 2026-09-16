"""Tests for lexical retrieval + eval metric math."""

from evaluation.eval_retrieval import coverage, mrr, recall_at_k


def test_recall_at_k():
    gold = {"a", "b"}
    ranked = ["x", "a", "y"]
    assert recall_at_k(gold, ranked, 1) == 0.0
    assert recall_at_k(gold, ranked, 2) == 1.0


def test_mrr():
    assert mrr({"a"}, ["x", "a", "y"]) == 0.5
    assert mrr({"a"}, ["a"]) == 1.0
    assert mrr({"a"}, ["x", "y"]) == 0.0


def test_coverage():
    assert coverage({"a", "b"}, ["a", "x", "b"], k=3) == 1.0
    assert coverage({"a", "b"}, ["a", "x"], k=3) == 0.5


def test_bm25_returns_gold_on_keyword_query():
    """Corpus-level sanity: a distinctive keyword query finds its chunk.

    Asserts top-10 presence — measured BM25 ranks this gold at ~6
    (keyword-dense FAQ chunks outrank it; that gap is why hybrid exists).
    """
    from ai.retrieval.bm25 import BM25Index
    from ai.retrieval.store import ChunkStore

    store = ChunkStore()
    if len(store) == 0:
        return  # corpus not built in this environment
    idx = BM25Index(store)
    hits = idx.search("income tax PM-KISAN exclusion", k=10, scheme="pmkisan")
    assert hits, "no hits returned"
    assert "pmkisan:operational_guidelines_2019:0002" in {cid for cid, _ in hits}
