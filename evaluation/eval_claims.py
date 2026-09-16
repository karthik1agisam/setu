#!/usr/bin/env python3
"""Claim-level verification eval: score baseline answers' unsupported rate.

Reads experiments/results/baseline_rag_outputs.jsonl (B3 arm), extracts
claims, verifies each against the chunks that were retrieved for it.

Usage:  uv run python evaluation/eval_claims.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ai.reasoning.llm import is_available
from ai.retrieval.store import ChunkStore
from ai.verification.claims import extract_claims
from ai.verification.verify import verify_claims

IN = Path("experiments/results/baseline_rag_outputs.jsonl")
OUT = Path("experiments/results/claims_baseline.json")


def main() -> int:
    if not is_available():
        print("Ollama not running (needed for claim extraction)")
        return 1
    if not IN.exists():
        print("run scripts/run_baseline.py first")
        return 1

    store = ChunkStore()
    rows = []
    for line in IN.open():
        rec = json.loads(line)
        ev = [store.get(cid) for cid in rec["retrieved"] if cid in store._by_id]
        cl = extract_claims(rec["answer"])
        vr = verify_claims(cl.claims, ev)
        rows.append(
            {
                "qid": rec["qid"],
                "answer": rec["answer"],
                "n_claims": len(vr.claims),
                "supported": vr.n_supported,
                "unsupported": vr.n_unsupported,
                "contradicted": vr.n_contradicted,
                "unsupported_rate": vr.unsupported_rate,
                "claims": [vars(c) for c in vr.claims],
            }
        )
        print(
            f"{rec['qid']}: {len(vr.claims)} claims | "
            f"supported={vr.n_supported} unsupported={vr.n_unsupported} "
            f"contradicted={vr.n_contradicted}"
        )

    total_claims = sum(r["n_claims"] for r in rows)
    total_bad = sum(r["unsupported"] + r["contradicted"] for r in rows)
    summary = {
        "n_answers": len(rows),
        "total_claims": total_claims,
        "unsupported_or_contradicted": total_bad,
        "unsupported_claim_rate": total_bad / total_claims if total_claims else 0.0,
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nunsupported-claim rate (B3 baseline): {summary['unsupported_claim_rate']:.3f}")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
