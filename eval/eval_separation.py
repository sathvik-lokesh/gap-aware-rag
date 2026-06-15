"""TIER A — embedding-level gap separation (no LLM, runs on all questions).

Question: using only the calibrated retrieval signals, how well can we tell an
ANSWERABLE question from an UNANSWERABLE one? We score every question, compute
the ROC-AUC of the gap signal predicting answerability, and record how the
ANSWERABLE/PARTIAL/GAP verdicts fall across the two classes.

Run:  uv run python eval/eval_separation.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gaprag import Index, retrieve, assess, Verdict  # noqa: E402

INDEX = ROOT / "index_store_squad"
QFILE = ROOT / "eval" / "questions.json"
OUT = ROOT / "eval" / "results"


def main():
    idx = Index.load(INDEX)
    questions = json.loads(QFILE.read_text())

    rows = []
    for q in questions:
        res = retrieve(idx, q["question"])
        a = assess(idx, res)
        rows.append({
            "label": q["label"],
            "answerable": int(q["label"] == "answerable"),
            "top1": res.signals.top1,
            "top1_pct": a.top1_percentile,
            "density": res.signals.density,
            "score_gap": res.signals.score_gap,
            "verdict": a.verdict.value,
        })

    y = np.array([r["answerable"] for r in rows])
    top1 = np.array([r["top1"] for r in rows])
    auc = roc_auc_score(y, top1)
    fpr, tpr, thr = roc_curve(y, top1)

    # Combined signal: 5-fold cross-validated logistic regression over all
    # coverage signals (out-of-fold scores -> no leakage).
    X = np.array([[r["top1"], r["score_gap"], r["density"]] for r in rows])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    oof = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    auc_combined = roc_auc_score(y, oof)
    fpr_c, tpr_c, _ = roc_curve(y, oof)

    # Verdict breakdown per class.
    def frac(label, verdict):
        sub = [r for r in rows if r["label"] == label]
        return sum(r["verdict"] == verdict for r in sub) / max(len(sub), 1)

    print(f"questions: {len(rows)}  "
          f"(answerable={int(y.sum())}, unanswerable={int((1-y).sum())})")
    print(f"\nROC-AUC, top1 similarity only      : {auc:.3f}")
    print(f"ROC-AUC, combined signals (5-fold CV): {auc_combined:.3f}")
    print("\nverdict distribution:")
    print(f"{'':14}{'ANSWERABLE':>12}{'PARTIAL':>10}{'GAP':>8}")
    for lab in ("answerable", "unanswerable"):
        print(f"{lab:14}"
              f"{frac(lab,'ANSWERABLE'):>12.0%}"
              f"{frac(lab,'PARTIAL'):>10.0%}"
              f"{frac(lab,'GAP'):>8.0%}")

    # Embedding-layer abstention prior: treat GAP as 'abstain'.
    gap_unans = frac("unanswerable", "GAP")
    gap_ans = frac("answerable", "GAP")
    print(f"\nGAP verdict flags {gap_unans:.0%} of unanswerable questions "
          f"(correct abstain prior)")
    print(f"GAP verdict wrongly flags {gap_ans:.0%} of answerable questions "
          f"(rescued later by LLM verification)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "separation.json").write_text(json.dumps({
        "auc_top1": auc, "auc_combined": auc_combined, "n": len(rows),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "fpr_combined": fpr_c.tolist(), "tpr_combined": tpr_c.tolist(),
        "rows": rows,
    }, indent=2))
    print(f"\nsaved -> {OUT/'separation.json'}")


if __name__ == "__main__":
    main()
