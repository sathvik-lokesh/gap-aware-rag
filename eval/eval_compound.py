"""HARDER eval — COMPOUND questions where decomposition + per-sub-question
verification should beat a single abstain-prompt baseline.

  multihop : score = both gold facts present (decomposition retrieves each part
             separately; single-shot retrieval often misses one).
  partial  : score = "graceful" = known fact present AND the missing part is
             acknowledged (answer the part you can, name the gap). A single
             abstain prompt is all-or-nothing and cannot do this.

Compares the same three systems as Tier B: naive (always answers), the FAIR
abstain-prompt baseline, and the Gap-Aware agent. Streams to results/compound.jsonl
(resumable). Paced for the Groq free-tier 12k-tokens/min cap.

Run:  GAPRAG_LLM_BACKEND=groq uv run python eval/eval_compound.py [aggregate]
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gaprag import Index, run as agent_run  # noqa: E402
from eval.eval_endtoend import naive_rag, abstain_rag, INDEX  # noqa: E402
from eval.metrics import contains_gold, looks_like_abstention  # noqa: E402

QFILE = ROOT / "eval" / "compound_questions.json"
OUT = ROOT / "eval" / "results"
JSONL = OUT / "compound.jsonl"


def score(kind, answer, q):
    """Return the per-system fields for one answer string."""
    if kind == "multihop":
        g1 = contains_gold(answer, q["gold1"])
        g2 = contains_gold(answer, q["gold2"])
        return {"both_present": g1 and g2, "one_present": g1 or g2}
    # partial
    known = contains_gold(answer, q["ans_gold"])
    flagged = looks_like_abstention(answer)
    return {"got_known": known, "flagged_gap": flagged,
            "graceful": known and flagged}


def done() -> set[str]:
    if not JSONL.exists():
        return set()
    return {json.loads(l)["question"] for l in JSONL.read_text().splitlines() if l}


def main():
    idx = Index.load(INDEX)
    qs = json.loads(QFILE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    already = done()
    print(f"{len(qs)} compound questions, {len(already)} cached")

    with JSONL.open("a") as f:
        for i, q in enumerate(qs, 1):
            if q["question"] in already:
                continue
            try:
                answers = {"naive": naive_rag(idx, q["question"]),
                           "abstain": abstain_rag(idx, q["question"]),
                           "gap_aware": agent_run(idx, q["question"]).final_answer}
            except Exception as e:  # noqa: BLE001  Groq 429 etc. -> save & resume
                print(f"stopped at [{i}/{len(qs)}]: {e}\nrerun to resume.")
                break
            rec = {"question": q["question"], "kind": q["kind"]}
            for sysname, ans in answers.items():
                rec[sysname] = {"answer": ans, **score(q["kind"], ans, q)}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{i}/{len(qs)}] {q['kind'][:5]} | {q['question'][:55]}")
            time.sleep(8)  # pace well under the Groq 12k-tokens/min cap

    aggregate()


def aggregate():
    recs = [json.loads(l) for l in JSONL.read_text().splitlines() if l]
    mh = [r for r in recs if r["kind"] == "multihop"]
    pa = [r for r in recs if r["kind"] == "partial"]
    systems = ["naive", "abstain", "gap_aware"]

    def rate(xs, sysname, k):
        return sum(r[sysname][k] for r in xs) / max(len(xs), 1)

    summary = {
        "n_multihop": len(mh), "n_partial": len(pa), "systems": systems,
        "multihop_both_present": {s: rate(mh, s, "both_present") for s in systems},
        "partial_graceful": {s: rate(pa, s, "graceful") for s in systems},
        "partial_got_known": {s: rate(pa, s, "got_known") for s in systems},
        "partial_flagged_gap": {s: rate(pa, s, "flagged_gap") for s in systems},
    }
    (OUT / "compound_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n========== HARDER EVAL: compound questions ==========")
    print(f"{'metric':34}" + "".join(f"{s:>11}" for s in systems))
    for label, key in [
        (f"multihop both-facts (n={len(mh)})", "multihop_both_present"),
        (f"partial graceful (n={len(pa)})", "partial_graceful"),
        ("  partial: got known fact", "partial_got_known"),
        ("  partial: flagged the gap", "partial_flagged_gap"),
    ]:
        print(f"{label:34}" + "".join(f"{summary[key][s]:>11.0%}" for s in systems))
    print(f"\nsaved -> {OUT/'compound_summary.json'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        aggregate()
    else:
        main()
