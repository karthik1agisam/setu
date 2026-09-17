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

### Measured baselines (n=12 seed set)

| Config | R@1 | R@5 | R@10 | MRR | Cov@10 |
|---|---|---|---|---|---|
| lexical (BM25) | 0.083 | 0.500 | 0.667 | 0.231 | 0.542 |
| dense (BGE-M3) | 0.167 | 0.583 | 0.667 | 0.360 | 0.500 |
| **hybrid (RRF)** | **0.250** | **0.583** | 0.667 | **0.380** | 0.542 |
| hybrid + rerank | 0.250 | 0.500 | 0.667 | 0.373 | 0.458 |

**Config chosen for pipeline: `hybrid`** — best R@1/MRR, no extra model pass.

### Findings (Phase 5)

1. **Context-prefixing (real fix):** exclusion clauses extracted as bare list
   items ("iii) All serving or retired officers…") carried no eligibility
   vocabulary and were unretrievable. Prepending the parent clause intro
   ("4.1 The following categories… shall NOT be eligible") fixed BM25 recall
   for these chunks. Implemented in `ai/ingestion/chunk.py`.
2. **Reranker did NOT help** (bge-reranker-v2-m3): gold exclusion chunk was
   in the candidate pool but scored below top-10 — bare list items score
   poorly vs prose even with context. Kept in code (`ai/retrieval/rerank.py`)
   but not the default config. Revisit on the larger benchmark set.
3. Residual misses (re02/re03/re05/re10) are exclusion-condition phrasing —
   candidate for LLM query expansion or structured-clause retrieval in a
   later phase.

Caveat: n=12 seed set → each item = 8.3 points of recall. Numbers will
stabilize when the full benchmark is authored.

## Claim-level verification (Phase 9)

NLI model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
(entailment / neutral / contradiction per claim vs each retrieved chunk).

```bash
uv run python scripts/calibrate_nli.py   # threshold sweep on labeled dev split
uv run python evaluation/eval_claims.py  # score baseline answers' claims
```

### Calibration (18 hand-labeled pairs, `data/benchmark/claims_dev.jsonl`)

- Best thresholds: entail≥0.50, contra≥0.50 → **accuracy 0.667**
- NLI is brittle on negated exclusions (flipped d03), numeric paraphrases
  (d01 ent=0.45), and reversed thresholds (d15 scored *entailed*).
- Consequence: NLI is a weak signal — the plan's deterministic numeric checks
  and rule-table extraction carry real weight. Honest limitation.

### Measured: baseline RAG (B3) unsupported-claim rate

| | value |
|---|---|
| answers scored | 12 |
| total claims | 35 |
| unsupported or contradicted | 17 |
| **unsupported-claim rate** | **0.486** |

Bounded by verifier accuracy (0.667) — some flags are NLI false-positives
(e.g., a correct free-treatment claim scored as contradicted). The rate is an
estimate, not ground truth; the verified-pipeline comparison is the
ablation's real test.

## Groundedness gate (Phase 10)

`ai/verification/gate.py` — release / clarify / abstain. Deterministic
numeric checker (`numeric.py`) parses Indian formats (lakh/crore/Rs.) and
flags claim numbers absent from evidence — catches what NLI misses.

Observed behavior (real run): correct `eligible` verdict (PMS-SC, ₹2L income)
→ gate **abstained** because NLI left 2/3 generated claims unsupported.
False abstention is the cost of a weak verifier — the gate prefers silence
over a possibly-wrong answer. **False-abstention rate is now a required
metric for the full benchmark** (n=12 seed set can't measure it reliably).
