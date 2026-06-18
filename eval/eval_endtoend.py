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

# The FAIR baseline: a single LLM call that is *allowed* to abstain — the cheap,
# strong baseline any reviewer asks for ("why not just prompt it to say IDK?").
# Gap-Aware RAG's calibration + verification has to beat THIS, not only the
# always-answer naive baseline above.
_ABSTAIN_SYS = (
    "Answer the question using ONLY the context below. If the context does not "
    "contain the answer, reply exactly: I don't know. Otherwise give a short, "
    "direct answer.")


def _ctx(idx: Index, question: str) -> str:
    hits = retrieve(idx, question, top_k=CFG.top_k).hits[:4]
    return "\n\n".join(h.chunk.text for h in hits)


def naive_rag(idx: Index, question: str) -> str:
    return chat(f"Context:\n{_ctx(idx, question)}\n\nQuestion: {question}",
                system=_NAIVE_SYS, model=AGENT_MODEL)


def abstain_rag(idx: Index, question: str) -> str:
    return chat(f"Context:\n{_ctx(idx, question)}\n\nQuestion: {question}",
                system=_ABSTAIN_SYS, model=AGENT_MODEL)


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
            abst = abstain_rag(idx, q["question"])
            ga = agent_run(idx, q["question"])
            rec = {
                "question": q["question"], "label": q["label"], "gold": q["gold"],
                "naive_answer": naive,
                "naive_abstained": looks_like_abstention(naive),
                "abstain_answer": abst,
                "abstain_abstained": looks_like_abstention(abst),
                "ga_answer": ga.final_answer, "ga_abstained": ga.abstained,
            }
            if q["label"] == "answerable":
                rec["naive_correct"] = (not rec["naive_abstained"]
                                        and contains_gold(naive, q["gold"]))
                rec["abstain_correct"] = (not rec["abstain_abstained"]
                                          and contains_gold(abst, q["gold"]))
                rec["ga_correct"] = (not ga.abstained
                                     and contains_gold(ga.final_answer, q["gold"]))
                rec["naive_f1"] = token_f1(naive, q["gold"])
                rec["abstain_f1"] = token_f1(abst, q["gold"])
                rec["ga_f1"] = token_f1(ga.final_answer, q["gold"])
            else:  # unanswerable -> any confident answer is a hallucination
                rec["naive_hallucinated"] = not rec["naive_abstained"]
                rec["abstain_hallucinated"] = not rec["abstain_abstained"]
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

    # Systems: always include naive + gap_aware; include the fair "abstain"
    # baseline only if every record carries it (older runs predate it).
    systems = [("naive", ""), ("gap_aware", "ga_")]
    if all("abstain_correct" in r or "abstain_hallucinated" in r for r in recs):
        systems.insert(1, ("abstain", "abstain_"))

    def acc_key(p): return f"{p}correct" if p else "naive_correct"
    def f1_key(p): return f"{p}f1" if p else "naive_f1"
    def hall_key(p): return f"{p}hallucinated" if p else "naive_hallucinated"

    summary = {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "systems": [s for s, _ in systems],
        "answerable_accuracy": {s: pct(ans, acc_key(p)) for s, p in systems},
        "answerable_f1": {s: pct(ans, f1_key(p)) for s, p in systems},
        "unanswerable_hallucination_rate": {s: pct(una, hall_key(p)) for s, p in systems},
        "unanswerable_abstention_rate": {s: 1 - pct(una, hall_key(p)) for s, p in systems},
    }
    (OUT / "endtoend_summary.json").write_text(json.dumps(summary, indent=2))

    cols = summary["systems"]
    print("\n============= GAP-AWARE vs BASELINE RAGs =============")
    header = f"{'metric':36}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    for label, sec, fmt in [
        ("answerable accuracy", "answerable_accuracy", "{:>12.0%}"),
        ("answerable token-F1", "answerable_f1", "{:>12.2f}"),
        ("unanswerable HALLUCINATION", "unanswerable_hallucination_rate", "{:>12.0%}"),
        ("unanswerable abstention (correct)", "unanswerable_abstention_rate", "{:>12.0%}"),
    ]:
        row = f"{label:36}" + "".join(fmt.format(summary[sec][c]) for c in cols)
        print(row)
    print(f"\nsaved -> {OUT/'endtoend_summary.json'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        aggregate()
    else:
        main()
