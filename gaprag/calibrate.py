"""Build a realistic 'answerable query' distribution.

Re-embedding a whole chunk as a query is unrealistically easy — full
paragraphs match each other far more strongly than a short question ever
matches its answer. So we ask a small LLM to write the kind of SHORT
questions a user would actually type, then measure their top1 scores against
the corpus. That distribution is the honest yardstick.

We only need enough pseudo-queries to estimate percentiles, so on a large
corpus we calibrate from a random SAMPLE of chunks — one LLM call per sampled
chunk, not per chunk in the whole corpus.
"""
from __future__ import annotations
import random
import numpy as np

from .config import FAST_MODEL, CALIBRATION_SAMPLE
from .index import Chunk
from .llm import chat
from .embeddings import embed_queries

_SYS = ("You write a single short, natural question that the given passage "
        "directly answers. Output ONLY the question, no preamble, under 15 words.")


def generate_pseudo_queries(chunks: list[Chunk]) -> list[str]:
    qs: list[str] = []
    for c in chunks:
        try:
            q = chat(c.text, system=_SYS, model=FAST_MODEL).splitlines()[0].strip()
        except Exception:
            q = ""
        if q:
            qs.append(q)
    return qs


def answerable_query_vectors(chunks: list[Chunk],
                             sample: int = CALIBRATION_SAMPLE) -> np.ndarray:
    """Return embedded, query-prefixed vectors for realistic pseudo-queries,
    sampling at most `sample` chunks so calibration scales to big corpora."""
    pool = chunks
    if len(chunks) > sample:
        pool = random.Random(0).sample(chunks, sample)
    pqs = generate_pseudo_queries(pool)
    if not pqs:
        raise RuntimeError("pseudo-query generation produced nothing")
    return embed_queries(pqs)
