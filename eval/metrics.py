"""Scoring utilities for the evaluation (SQuAD-style normalization)."""
from __future__ import annotations
import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)

# Phrases a model uses when it (correctly or not) declines to answer.
_ABSTAIN = re.compile(
    r"\b(not? (mention|state|provid|specif|given|available|include|contain)"
    r"|no (information|answer|mention|data)|cannot (be )?(determin|answer|found)"
    r"|can'?t (determin|answer)|does not (say|mention|provide|specify)"
    r"|doesn'?t (say|mention|provide|specify)|unanswerable|not (found|present)"
    r"|insufficient|unknown|n/?a|unclear|unable to)\b",
    re.IGNORECASE,
)


def normalize(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def contains_gold(pred: str, golds: list[str]) -> bool:
    """True if any gold answer appears (normalized) within the prediction."""
    npred = normalize(pred)
    return any(normalize(g) and normalize(g) in npred for g in golds)


def token_f1(pred: str, golds: list[str]) -> float:
    """Max SQuAD token-F1 of the prediction against the gold answers."""
    best = 0.0
    pt = normalize(pred).split()
    for g in golds:
        gt = normalize(g).split()
        if not pt or not gt:
            continue
        common = _overlap(pt, gt)
        if common == 0:
            continue
        prec, rec = common / len(pt), common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _overlap(a: list[str], b: list[str]) -> int:
    from collections import Counter
    c = Counter(a) & Counter(b)
    return sum(c.values())


def looks_like_abstention(text: str) -> bool:
    return bool(_ABSTAIN.search(text or ""))
