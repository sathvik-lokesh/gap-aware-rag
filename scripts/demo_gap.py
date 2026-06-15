"""M1+M2 demo: build the index, then watch the gap detector separate
answerable questions from ones that fall outside the corpus.

Run:  uv run python scripts/demo_gap.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaprag import build_index, retrieve, assess  # noqa: E402

# Facts that ARE in the corpus -> should be ANSWERABLE
ANSWERABLE = [
    "What temperature can Aurelia's robots operate at?",
    "How long does the Frostpath F2 battery last?",
    "Who is the CEO of Aurelia Robotics?",
    "How often should the heated treads be inspected?",
]

# Facts deliberately NOT in the corpus -> should be GAP (or at least PARTIAL)
GAPS = [
    "What is Aurelia Robotics' annual revenue?",
    "Does the Frostpath work in hot desert warehouses?",
    "Who is Aurelia's Chief Financial Officer?",
    "What is the airspeed velocity of an unladen swallow?",
]


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    banner("BUILDING + CALIBRATING INDEX")
    idx = build_index()
    idx.save()
    print(f"chunks: {len(idx.chunks)}  |  dim: {idx.vectors.shape[1]}")
    print("corpus nearest-neighbor similarity percentiles:")
    for p, v in idx.nn_percentiles.items():
        print(f"   p{p:<2} = {v:.3f}")
    print("  ^ this distribution is the yardstick. A query whose best match")
    print("    sits low in it is landing in empty space = a knowledge gap.")

    for title, queries in [("ANSWERABLE QUERIES", ANSWERABLE),
                           ("OUT-OF-CORPUS QUERIES (expected gaps)", GAPS)]:
        banner(title)
        for q in queries:
            res = retrieve(idx, q)
            a = assess(idx, res)
            print(f"\nQ: {q}")
            print(f"   {a}")
            print(f"   best chunk: \"{res.hits[0].chunk.text[:80]}...\"")


if __name__ == "__main__":
    main()
