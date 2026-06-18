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
# colour + label per system id (the fair abstain baseline sits between the two)
SYSTEMS = {
    "naive": ("#e07b39", "naive RAG (always answers)"),
    "abstain": ("#8c9bab", "RAG + abstain prompt"),
    "gap_aware": ("#1f3a5f", "Gap-Aware RAG"),
}


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
    metrics = [("Answerable\naccuracy ↑", "answerable_accuracy"),
               ("Unanswerable\nHALLUCINATION ↓", "unanswerable_hallucination_rate"),
               ("Unanswerable\nabstention ↑", "unanswerable_abstention_rate")]
    sys_ids = s.get("systems", ["naive", "gap_aware"])

    x = np.arange(len(metrics))
    w = 0.8 / len(sys_ids)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for j, sid in enumerate(sys_ids):
        color, label = SYSTEMS[sid]
        vals = [s[sec][sid] for _, sec in metrics]
        off = (j - (len(sys_ids) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, label=label, color=color)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([m for m, _ in metrics])
    ax.set_ylim(0, 1); ax.set_ylabel("rate")
    ax.set_title(f"Gap-Aware vs baseline RAGs  (SQuAD 2.0, "
                 f"n={s['n_answerable']+s['n_unanswerable']})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_endtoend.png", dpi=130)
    print("wrote fig_endtoend.png")


def plot_compound():
    s = json.loads((OUT / "compound_summary.json").read_text())
    metrics = [(f"Multi-hop\nboth facts found ↑\n(n={s['n_multihop']})",
                "multihop_both_present"),
               (f"Partial coverage\ngraceful answer ↑\n(n={s['n_partial']})",
                "partial_graceful")]
    sys_ids = s.get("systems", ["naive", "abstain", "gap_aware"])

    x = np.arange(len(metrics)); w = 0.8 / len(sys_ids)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for j, sid in enumerate(sys_ids):
        color, label = SYSTEMS[sid]
        vals = [s[sec][sid] for _, sec in metrics]
        off = (j - (len(sys_ids) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, label=label, color=color)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.0%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([m for m, _ in metrics])
    ax.set_ylim(0, 1); ax.set_ylabel("rate")
    ax.set_title("Where Gap-Aware RAG earns its keep: COMPOUND questions")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_compound.png", dpi=130)
    print("wrote fig_compound.png")


if __name__ == "__main__":
    if (OUT / "separation.json").exists():
        plot_separation()
    if (OUT / "endtoend_summary.json").exists():
        plot_endtoend()
    if (OUT / "compound_summary.json").exists():
        plot_compound()
