"""THE NOVEL CORE.

Turns coverage signals into a calibrated answerability verdict. The key
move: we judge the query's best match against the corpus's OWN
nearest-neighbor similarity distribution (computed at index time), not
against a magic constant. 'Good enough' is defined relative to this corpus.

Three verdicts:
  ANSWERABLE — query lands in a well-supported region; answer it.
  PARTIAL    — some support but thin/scattered; needs multi-hop or hedging.
  GAP        — query lands in empty embedding space; abstain and NAME the gap.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from .config import GAP
from .index import Index
from .retriever import RetrievalResult


class Verdict(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    PARTIAL = "PARTIAL"
    GAP = "GAP"


@dataclass
class GapAssessment:
    verdict: Verdict
    top1_percentile: float   # where top1 sits in the corpus NN distribution
    reason: str
    signals: dict

    def __str__(self) -> str:
        return (f"[{self.verdict}] top1 at {self.top1_percentile:.0f}th pct "
                f"| {self.reason} | {self.signals}")


def assess(index: Index, result: RetrievalResult) -> GapAssessment:
    sig = result.signals
    pct = index.sim_to_percentile(sig.top1)

    dense_enough = sig.density >= GAP.min_dense_neighbors

    if pct < GAP.gap_percentile and not dense_enough:
        verdict = Verdict.GAP
        reason = ("best match weaker than ~75% of in-corpus neighbors and "
                  "query sits in sparse embedding space — likely no coverage")
    elif pct >= GAP.answerable_percentile and dense_enough:
        verdict = Verdict.ANSWERABLE
        reason = "strong, well-supported match"
    else:
        verdict = Verdict.PARTIAL
        reason = ("some support but thin, scattered, or borderline — "
                  "hedge or go multi-hop")

    return GapAssessment(
        verdict=verdict,
        top1_percentile=pct,
        reason=reason,
        signals=sig.as_dict(),
    )
