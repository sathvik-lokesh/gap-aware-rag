"""Build a HARDER eval set of COMPOUND questions from the labeled SQuAD set.

Two kinds, both designed to test what a single abstain-prompt baseline cannot do
but the decompose -> per-sub-question verify agent can:

  multihop : two ANSWERABLE facts from the same article but DIFFERENT paragraphs
             (context_id). Single-shot top-k retrieval often grabs chunks for one
             part and misses the other; decomposition retrieves each separately.
             Scored: both gold facts present in the answer.

  partial  : one ANSWERABLE + one genuinely UNANSWERABLE part from the same
             article. The right behaviour is to answer the known part AND name
             the missing one. An abstain prompt is all-or-nothing (abstains on
             everything, losing the known part, or answers and fabricates the
             missing part). Scored: known fact present AND gap acknowledged.

Run:  uv run python eval/build_compound.py [N_PER_KIND]
"""
from __future__ import annotations
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QFILE = ROOT / "eval" / "questions.json"
OUT = ROOT / "eval" / "compound_questions.json"


def _join(q1: str, q2: str) -> str:
    """Two standalone questions -> one compound (keeps both '?' so the agent's
    compound heuristic fires, and reads naturally)."""
    q2 = q2[0].lower() + q2[1:] if q2 else q2
    return f"{q1.rstrip()} And {q2}"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    qs = json.loads(QFILE.read_text())
    rng = random.Random(7)

    by_topic_ans = defaultdict(list)
    by_topic_una = defaultdict(list)
    for q in qs:
        if q["label"] == "answerable" and q["gold"]:
            by_topic_ans[q["topic"]].append(q)
        elif q["label"] == "unanswerable":
            by_topic_una[q["topic"]].append(q)

    multihop, partial = [], []

    # multihop: two answerable, same topic, DIFFERENT source paragraph
    for topic, items in by_topic_ans.items():
        rng.shuffle(items)
        used = set()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["context_id"] == b["context_id"]:
                    continue
                if a["context_id"] in used or b["context_id"] in used:
                    continue
                used.update({a["context_id"], b["context_id"]})
                multihop.append({
                    "kind": "multihop", "topic": topic,
                    "question": _join(a["question"], b["question"]),
                    "gold1": a["gold"], "gold2": b["gold"],
                })
                break

    # partial: one answerable + one unanswerable, same topic
    for topic, items in by_topic_ans.items():
        unas = by_topic_una.get(topic, [])
        if not unas:
            continue
        rng.shuffle(items); rng.shuffle(unas)
        for a, u in zip(items, unas):
            partial.append({
                "kind": "partial", "topic": topic,
                "question": _join(a["question"], u["question"]),
                "ans_gold": a["gold"], "absent_part": u["question"],
            })

    rng.shuffle(multihop); rng.shuffle(partial)
    out = multihop[:n] + partial[:n]
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} compound questions "
          f"({min(n, len(multihop))} multihop + {min(n, len(partial))} partial) "
          f"-> {OUT}")
    for r in out[:4]:
        print(f"  [{r['kind']}] {r['question'][:90]}")


if __name__ == "__main__":
    main()
