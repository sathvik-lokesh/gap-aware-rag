"""TIER B — end-to-end Gap-Aware RAG vs a naive RAG baseline (uses the LLM).

Naive RAG: retrieve top-k, answer from context with no notion of gaps (the
common minimal implementation). It will answer even unanswerable questions.
Gap-Aware RAG: the full agent — abstains and names gaps.

Metrics:
  answerable   -> accuracy (gold answer present in the response)
  unanswerable -> HALLUCINATION rate (gave a confident answer instead of
                  abstaining) and abstention accuracy.

Results stream to results/endtoend.jsonl so the run is resumable on this
constrained box (a timeout/OOM mid-run loses nothing; just rerun).

Run:  uv run python eval/eval_endtoend.py [N_PER_LABEL]
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gaprag import Index, retrieve, run as agent_run  # noqa: E402
from gaprag.llm import chat  # noqa: E402
from gaprag.agent import AGENT_MODEL  # noqa: E402
from gaprag.config import OLLAMA_URL, CFG, LLM_BACKEND  # noqa: E402
from eval.metrics import contains_gold, token_f1, looks_like_abstention  # noqa: E402

INDEX = ROOT / "index_store_squad"
QFILE = ROOT / "eval" / "questions.json"
OUT = ROOT / "eval" / "results"
JSONL = OUT / "endtoend.jsonl"

_NAIVE_SYS = ("Answer the question using the context below. Give a short, direct "
              "answer.")


def naive_rag(idx: Index, question: str) -> str:
    hits = retrieve(idx, question, top_k=CFG.top_k).hits[:4]
    ctx = "\n\n".join(h.chunk.text for h in hits)
    return chat(f"Context:\n{ctx}\n\nQuestion: {question}", system=_NAIVE_SYS,
                model=AGENT_MODEL)


def warm_up():
    # Only meaningful for the local Ollama backend (7b cold-starts slowly on the
    # 8GB box). On Groq the weights are always warm, so skip it — otherwise we'd
    # needlessly load 7b into local RAM.
    if LLM_BACKEND == "groq":
        return
    requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": AGENT_MODEL, "messages": [{"role": "user", "content": "ok"}],
        "stream": False, "keep_alive": "10m"}, timeout=300)


def sample(questions, n_per_label):
    ans = [q for q in questions if q["label"] == "answerable"]
    una = [q for q in questions if q["label"] == "unanswerable"]
    rng = random.Random(13)
    rng.shuffle(ans); rng.shuffle(una)
    return ans[:n_per_label] + una[:n_per_label]


def done_questions() -> set[str]:
    if not JSONL.exists():
        return set()
    return {json.loads(l)["question"] for l in JSONL.read_text().splitlines() if l}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    idx = Index.load(INDEX)
    qs = sample(json.loads(QFILE.read_text()), n)
    OUT.mkdir(parents=True, exist_ok=True)
    already = done_questions()
    print(f"evaluating {len(qs)} questions ({n}/label), {len(already)} cached")
    warm_up()

    with JSONL.open("a") as f:
        for i, q in enumerate(qs, 1):
            if q["question"] in already:
                continue
            naive = naive_rag(idx, q["question"])
            ga = agent_run(idx, q["question"])
            rec = {
                "question": q["question"], "label": q["label"], "gold": q["gold"],
                "naive_answer": naive,
                "naive_abstained": looks_like_abstention(naive),
                "ga_answer": ga.final_answer, "ga_abstained": ga.abstained,
            }
            if q["label"] == "answerable":
                rec["naive_correct"] = (not rec["naive_abstained"]
                                        and contains_gold(naive, q["gold"]))
                rec["ga_correct"] = (not ga.abstained
                                     and contains_gold(ga.final_answer, q["gold"]))
                rec["naive_f1"] = token_f1(naive, q["gold"])
                rec["ga_f1"] = token_f1(ga.final_answer, q["gold"])
            else:  # unanswerable -> any confident answer is a hallucination
                rec["naive_hallucinated"] = not rec["naive_abstained"]
                rec["ga_hallucinated"] = not ga.abstained
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{i}/{len(qs)}] {q['label'][:4]} | {q['question'][:55]}")

    aggregate()


def aggregate():
    recs = [json.loads(l) for l in JSONL.read_text().splitlines() if l]
    ans = [r for r in recs if r["label"] == "answerable"]
    una = [r for r in recs if r["label"] == "unanswerable"]

    def pct(xs, k):
        return sum(r[k] for r in xs) / max(len(xs), 1)

    summary = {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "answerable_accuracy": {"naive": pct(ans, "naive_correct"),
                                "gap_aware": pct(ans, "ga_correct")},
        "answerable_f1": {"naive": pct(ans, "naive_f1"),
                          "gap_aware": pct(ans, "ga_f1")},
        "unanswerable_hallucination_rate": {"naive": pct(una, "naive_hallucinated"),
                                            "gap_aware": pct(una, "ga_hallucinated")},
        "unanswerable_abstention_rate": {
            "naive": 1 - pct(una, "naive_hallucinated"),
            "gap_aware": 1 - pct(una, "ga_hallucinated")},
    }
    (OUT / "endtoend_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ GAP-AWARE vs NAIVE RAG ================")
    print(f"{'metric':38}{'naive':>10}{'gap-aware':>12}")
    print(f"{'answerable accuracy':38}"
          f"{summary['answerable_accuracy']['naive']:>10.0%}"
          f"{summary['answerable_accuracy']['gap_aware']:>12.0%}")
    print(f"{'unanswerable HALLUCINATION rate':38}"
          f"{summary['unanswerable_hallucination_rate']['naive']:>10.0%}"
          f"{summary['unanswerable_hallucination_rate']['gap_aware']:>12.0%}")
    print(f"{'unanswerable abstention (correct)':38}"
          f"{summary['unanswerable_abstention_rate']['naive']:>10.0%}"
          f"{summary['unanswerable_abstention_rate']['gap_aware']:>12.0%}")
    print(f"\nsaved -> {OUT/'endtoend_summary.json'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        aggregate()
    else:
        main()
