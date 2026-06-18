"""Backfill the FAIR 'prompted-to-abstain' baseline onto an existing
results/endtoend.jsonl without re-running the (expensive) gap-aware agent.

For each cached record it makes ONE extra LLM call (abstain-prompted RAG over the
same retrieved context) and adds the abstain_* fields, then re-aggregates. Safe
to re-run: records that already have the baseline are skipped.

Run:  GAPRAG_LLM_BACKEND=groq uv run python eval/augment_abstain.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gaprag import Index  # noqa: E402
from eval.eval_endtoend import abstain_rag, aggregate, INDEX, JSONL  # noqa: E402
from eval.metrics import contains_gold, token_f1, looks_like_abstention  # noqa: E402


def main():
    recs = [json.loads(l) for l in JSONL.read_text().splitlines() if l]
    idx = Index.load(INDEX)
    todo = [r for r in recs if "abstain_answer" not in r]
    print(f"{len(recs)} records, {len(todo)} need the abstain baseline")

    def flush():  # checkpoint after every record -> crash-safe & resumable
        JSONL.write_text("".join(json.dumps(r) + "\n" for r in recs))

    for i, r in enumerate(recs, 1):
        if "abstain_answer" in r:
            continue
        try:
            abst = abstain_rag(idx, r["question"])
        except Exception as e:  # noqa: BLE001  (Groq 429 etc.) -> save & resume
            print(f"stopped at [{i}/{len(recs)}]: {e}\nrerun to resume.")
            flush()
            aggregate()
            return
        r["abstain_answer"] = abst
        r["abstain_abstained"] = looks_like_abstention(abst)
        if r["label"] == "answerable":
            r["abstain_correct"] = (not r["abstain_abstained"]
                                    and contains_gold(abst, r["gold"]))
            r["abstain_f1"] = token_f1(abst, r["gold"])
        else:
            r["abstain_hallucinated"] = not r["abstain_abstained"]
        flush()
        print(f"[{i}/{len(recs)}] {r['label'][:4]} | {r['question'][:50]} "
              f"-> {'ABSTAIN' if r['abstain_abstained'] else 'answer'}")
        time.sleep(4)  # pace just under the Groq free-tier 12k tokens/min cap

    aggregate()


if __name__ == "__main__":
    main()
