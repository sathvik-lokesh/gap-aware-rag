"""M3 demo: the full gap-aware agent loop.

Shows the payoff of the verification layer:
  - "Who is the CEO?"  -> answered (passage states it)
  - "Who is the CFO?"  -> abstains (passage is topical but lacks the fact)
  - a multi-part question -> decomposed, each part answered or flagged a gap

Run:  uv run python scripts/demo_agent.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from gaprag import build_index, run, Index  # noqa: E402
from gaprag.config import INDEX_DIR, OLLAMA_URL  # noqa: E402
from gaprag.agent import AGENT_MODEL  # noqa: E402


def warm_up():
    """Load the agent model into RAM once so it stays resident (keep_alive)
    for the run — avoids a cold 4.7GB reload mid-loop on constrained boxes."""
    print(f"(warming up {AGENT_MODEL}...)")
    requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": AGENT_MODEL, "messages": [{"role": "user", "content": "ok"}],
        "stream": False, "keep_alive": "10m"}, timeout=300)

QUESTIONS = [
    "Who is the CEO of Aurelia Robotics?",        # fact present -> answer
    "Who is the CFO of Aurelia Robotics?",        # fact absent  -> abstain
    "What is Aurelia's annual revenue?",          # topical gap  -> abstain
    "How long does the F2 battery last and how heavy a payload can it lift?",  # multi-hop
    "What battery does the F2 use and what is Aurelia's stock price?",         # mixed: 1 fact + 1 gap
]


def load_or_build() -> Index:
    if (INDEX_DIR / "vectors.npy").exists() and (INDEX_DIR / "nn_sims.npy").exists():
        print(f"(loading cached index from {INDEX_DIR})")
        return Index.load()
    print("(building index...)")
    idx = build_index()
    idx.save()
    return idx


def main():
    idx = load_or_build()
    warm_up()
    for q in QUESTIONS:
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        r = run(idx, q)
        print("-" * 72)
        for s in r.sub_answers:
            tag = "✓ verified" if s.supported else f"✗ GAP ({s.gap_reason})"
            print(f"  [{s.verdict.value:10} p{s.top1_percentile:>3.0f}] {s.subq}")
            print(f"       -> {tag}")
            if s.supported:
                print(f"          answer: {s.answer}  [src: {s.source_doc}]")
        print("-" * 72)
        print(f"ANSWER{' (ABSTAINED)' if r.abstained else ''}:\n  {r.final_answer}")


if __name__ == "__main__":
    main()
