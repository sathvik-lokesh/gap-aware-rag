"""Render result plots for the README.

  results/fig_separation.png  -> gap-score distributions + ROC (Tier A)
  results/fig_endtoend.png    -> Gap-Aware vs naive RAG bars (Tier B)

Run:  uv run python eval/make_plots.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
NAVY, ORANGE = "#1f3a5f", "#e07b39"


def plot_separation():
    data = json.loads((OUT / "separation.json").read_text())
    rows = data["rows"]
    ans = [r["top1"] for r in rows if r["label"] == "answerable"]
    una = [r["top1"] for r in rows if r["label"] == "unanswerable"]

    fig, (axh, axr) = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.linspace(min(ans + una), max(ans + una), 30)
    axh.hist(ans, bins=bins, alpha=0.7, color=NAVY, label="answerable")
    axh.hist(una, bins=bins, alpha=0.7, color=ORANGE, label="unanswerable")
    axh.set_title("Top-1 retrieval similarity by question type")
    axh.set_xlabel("top-1 cosine similarity"); axh.set_ylabel("count")
    axh.legend()
    axh.text(0.02, 0.95, "overlap is the point:\nembeddings barely separate\n"
             "adversarial unanswerables",
             transform=axh.transAxes, va="top", fontsize=8, color="#555")

    axr.plot(data["fpr"], data["tpr"], color=ORANGE,
             label=f"top-1 only (AUC={data['auc_top1']:.2f})")
    axr.plot(data["fpr_combined"], data["tpr_combined"], color=NAVY,
             label=f"combined signals (AUC={data['auc_combined']:.2f})")
    axr.plot([0, 1], [0, 1], "--", color="#aaa")
    axr.set_title("Embedding-layer gap detection (ROC)")
    axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate")
    axr.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT / "fig_separation.png", dpi=130)
    print("wrote fig_separation.png")


def plot_endtoend():
    s = json.loads((OUT / "endtoend_summary.json").read_text())
    labels = ["Answerable\naccuracy ↑", "Unanswerable\nHALLUCINATION ↓",
              "Unanswerable\nabstention ↑"]
    naive = [s["answerable_accuracy"]["naive"],
             s["unanswerable_hallucination_rate"]["naive"],
             s["unanswerable_abstention_rate"]["naive"]]
    ga = [s["answerable_accuracy"]["gap_aware"],
          s["unanswerable_hallucination_rate"]["gap_aware"],
          s["unanswerable_abstention_rate"]["gap_aware"]]

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    b1 = ax.bar(x - w/2, naive, w, label="naive RAG", color=ORANGE)
    b2 = ax.bar(x + w/2, ga, w, label="Gap-Aware RAG", color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("rate")
    ax.set_title(f"Gap-Aware vs naive RAG  (SQuAD 2.0, "
                 f"n={s['n_answerable']+s['n_unanswerable']})")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_endtoend.png", dpi=130)
    print("wrote fig_endtoend.png")


if __name__ == "__main__":
    if (OUT / "separation.json").exists():
        plot_separation()
    if (OUT / "endtoend_summary.json").exists():
        plot_endtoend()
