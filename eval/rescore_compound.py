"""Re-score the cached compound answers with the current metrics (no LLM calls).

The eval stores each system's raw answer text, so when the scoring rules change
we can recompute the verdicts from disk instead of re-running the models. Joins
compound.jsonl (answers) with compound_questions.json (gold), recomputes every
system's score(), rewrites the jsonl, and re-aggregates.

Run:  uv run python eval/rescore_compound.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.eval_compound import score, aggregate, JSONL  # noqa: E402

QFILE = ROOT / "eval" / "compound_questions.json"


def main():
    gold = {q["question"]: q for q in json.loads(QFILE.read_text())}
    recs = [json.loads(l) for l in JSONL.read_text().splitlines() if l]
    for r in recs:
        q = gold[r["question"]]
        for sysname in ("naive", "abstain", "gap_aware"):
            ans = r[sysname]["answer"]
            r[sysname] = {"answer": ans, **score(r["kind"], ans, q)}
    JSONL.write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"re-scored {len(recs)} records")
    aggregate()


if __name__ == "__main__":
    main()
