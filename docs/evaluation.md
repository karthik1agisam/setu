# Evaluation

All metrics are produced by scripts in `evaluation/` and written to
`experiments/results/`. Never hand-edited.

## Retrieval

Gold standard: `data/benchmark/retrieval_eval.jsonl` — seed set of
document-grounded questions with `gold_chunk_ids` (expands during benchmark
construction phase).

```bash
make eval-retrieval CONFIG=lexical
```

Metrics: Recall@{1,3,5,10}, MRR, Coverage@10 (fraction of gold chunks in top-10).

### Measured baselines

| Config | R@1 | R@5 | R@10 | MRR | Cov@10 |
|---|---|---|---|---|---|
| lexical (BM25) | 0.083 | 0.500 | 0.667 | 0.231 | 0.542 |

Notable failure pattern: BM25 prefers FAQ chunks containing keywords over the
authoritative guideline clauses (re02, re03) — motivates semantic + hybrid
retrieval (Phase 5).
