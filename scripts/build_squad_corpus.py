"""Build a real RAG corpus + labeled gap-detection question set from SQuAD 2.0.

SQuAD 2.0 is ideal for this project: every topic mixes ANSWERABLE questions with
deliberately UNANSWERABLE ones (is_impossible) — questions that look on-topic but
whose answer is not in the passage. That is exactly the 'topical but not factual'
gap our system targets, with ground-truth labels.

Outputs:
  corpus_squad/        one .md file per paragraph (the retrieval corpus)
  eval/questions.json  [{question, label: answerable|unanswerable, gold:[...]}]

Run:  uv run python scripts/build_squad_corpus.py
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SQUAD = ROOT / "data_squad" / "dev-v2.0.json"
CORPUS_DIR = ROOT / "corpus_squad"
QFILE = ROOT / "eval" / "questions.json"

N_TOPICS = 4          # a few coherent topics -> a focused, real corpus
MAX_CONTEXTS = 110    # cap corpus size (keeps embedding/calibration tractable)
PER_LABEL = 160       # cap questions per label for the eval set


def main():
    data = json.loads(SQUAD.read_text())["data"]
    rng = random.Random(7)
    topics = rng.sample(data, N_TOPICS)

    CORPUS_DIR.mkdir(exist_ok=True)
    for old in CORPUS_DIR.glob("*.md"):
        old.unlink()

    contexts = []   # (cid, title, text)
    answerable, unanswerable = [], []

    for t in topics:
        title = t["title"]
        for pi, para in enumerate(t["paragraphs"]):
            if len(contexts) >= MAX_CONTEXTS:
                break
            cid = f"{title}__{pi}"
            contexts.append((cid, title, para["context"]))
            for qa in para["qas"]:
                rec = {"question": qa["question"].strip(), "context_id": cid,
                       "topic": title}
                if qa.get("is_impossible"):
                    rec["label"] = "unanswerable"
                    rec["gold"] = []
                    unanswerable.append(rec)
                else:
                    golds = sorted({a["text"] for a in qa["answers"] if a["text"]})
                    if not golds:
                        continue
                    rec["label"] = "answerable"
                    rec["gold"] = golds
                    answerable.append(rec)

    # Write corpus files.
    for cid, title, text in contexts:
        safe = cid.replace("/", "_").replace(" ", "_")
        (CORPUS_DIR / f"{safe}.md").write_text(f"# {title}\n\n{text}\n",
                                               encoding="utf-8")

    # Balanced, capped question set.
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    qs = answerable[:PER_LABEL] + unanswerable[:PER_LABEL]
    rng.shuffle(qs)
    QFILE.parent.mkdir(exist_ok=True)
    QFILE.write_text(json.dumps(qs, indent=2, ensure_ascii=False))

    print(f"topics      : {[t['title'] for t in topics]}")
    print(f"corpus files: {len(contexts)}  -> {CORPUS_DIR}")
    print(f"questions   : {len(qs)}  (answerable={min(len(answerable),PER_LABEL)},"
          f" unanswerable={min(len(unanswerable),PER_LABEL)})")
    print(f"            -> {QFILE}")


if __name__ == "__main__":
    main()
