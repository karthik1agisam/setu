#!/usr/bin/env python3
"""Retrieval evaluation harness.

Runs a retrieval config over data/benchmark/retrieval_eval.jsonl and reports:
  Recall@{1,3,5,10}  fraction of questions with ≥1 gold chunk in top-k
  MRR                mean reciprocal rank of first gold hit
  coverage           fraction of all gold chunks retrieved within k=10

Configs:
  lexical   BM25 only (this phase's baseline)
  (dense / hybrid / reranked plug in later phases)

Usage:
  uv run python evaluation/eval_retrieval.py --config lexical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai.retrieval.bm25 import BM25Index
from ai.retrieval.store import ChunkStore


def _build_search(config: str, store: ChunkStore):
    """Return search(query, k, scheme) -> [(chunk_id, score)] for a config."""
    if config == "lexical":
        bm25 = BM25Index(store)
        return lambda q, k, s: bm25.search(q, k=k, scheme=s)
    if config == "dense":
        from ai.retrieval.dense import DenseIndex

        dense = DenseIndex(store)
        return lambda q, k, s: dense.search(q, k=k)
    if config == "hybrid":
        from ai.retrieval.dense import DenseIndex
        from ai.retrieval.hybrid import HybridIndex

        hyb = HybridIndex(BM25Index(store), DenseIndex(store))
        return lambda q, k, s: hyb.search(q, k=k, scheme=s)
    if config == "hybrid_rerank":
        from ai.retrieval.dense import DenseIndex
        from ai.retrieval.hybrid import HybridIndex, rrf_fuse
        from ai.retrieval.rerank import Reranker

        bm25 = BM25Index(store)
        dense = DenseIndex(store)

        def search(q: str, k: int, s: str | None):
            pool = rrf_fuse(dense.search(q, k=15), bm25.search(q, k=15, scheme=s))[:15]
            return Reranker.rerank(q, pool, store, k=k)

        return search
    raise ValueError(f"unknown config {config}")

EVAL_FILE = Path("data/benchmark/retrieval_eval.jsonl")
RESULTS_DIR = Path("experiments/results")
KS = (1, 3, 5, 10)


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    return 1.0 if gold & set(ranked[:k]) else 0.0


def mrr(gold: set[str], ranked: list[str]) -> float:
    for i, cid in enumerate(ranked):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


def coverage(gold: set[str], ranked: list[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def run(config: str) -> dict:
    store = ChunkStore()
    search = _build_search(config, store)

    items = [json.loads(line) for line in EVAL_FILE.open()]
    per_item = []
    rec_sums = dict.fromkeys(KS, 0.0)
    mrr_sum = 0.0
    cov_sum = 0.0

    for it in items:
        gold = set(it["gold_chunk_ids"])
        hits = search(it["question"], max(KS), it["scheme"])
        ranked = [cid for cid, _ in hits]

        rec = {k: recall_at_k(gold, ranked, k) for k in KS}
        for k in KS:
            rec_sums[k] += rec[k]
        m = mrr(gold, ranked)
        cov = coverage(gold, ranked)
        mrr_sum += m
        cov_sum += cov
        per_item.append(
            {
                "qid": it["qid"],
                "top1": ranked[0] if ranked else None,
                "gold": sorted(gold),
                **{f"r@{k}": rec[k] for k in KS},
                "mrr": m,
                "coverage": cov,
            }
        )

    n = len(items)
    return {
        "config": config,
        "n": n,
        "recall_at": {k: rec_sums[k] / n for k in KS},
        "mrr": mrr_sum / n,
        "coverage@10": cov_sum / n,
        "per_item": per_item,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="lexical")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    res = run(args.config)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{args.config}.json"
    out.write_text(json.dumps(res, indent=2))

    print(f"config={res['config']} n={res['n']}")
    for k, v in res["recall_at"].items():
        print(f"  Recall@{k}: {v:.3f}")
    print(f"  MRR:        {res['mrr']:.3f}")
    print(f"  Coverage@10:{res['coverage@10']:.3f}")
    if args.verbose:
        for it in res["per_item"]:
            print(f"  {it['qid']}: top1={it['top1']} r@5={it['r@5']}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
