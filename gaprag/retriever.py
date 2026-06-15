"""Instrumented retriever. A normal RAG retriever returns top-k passages.
Ours ALSO returns 'coverage signals' describing how well the corpus actually
supports the query. Those signals feed the gap detector."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .config import CFG
from .embeddings import embed_query
from .index import Index, Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class CoverageSignals:
    top1: float            # strength of the single best match
    score_gap: float       # top1 minus mean of the rest -> clear winner or mush?
    density: int           # how many corpus chunks are "close" to the query
    concentration: float   # 0..1, is support coherent (one doc) or scattered?

    def as_dict(self) -> dict:
        return {
            "top1": round(self.top1, 4),
            "score_gap": round(self.score_gap, 4),
            "density": self.density,
            "concentration": round(self.concentration, 3),
        }


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    signals: CoverageSignals


def retrieve(index: Index, query: str, top_k: int = CFG.top_k) -> RetrievalResult:
    q = embed_query(query)                    # query-prefixed, normalized
    sims = index.vectors @ q                  # cosine to every chunk
    order = np.argsort(-sims)[:top_k]
    hits = [Hit(chunk=index.chunks[i], score=float(sims[i])) for i in order]

    top1 = float(sims[order[0]])
    rest = sims[order[1:]]
    score_gap = top1 - float(rest.mean()) if rest.size else 0.0

    # density: count corpus chunks at least as similar as the corpus MEDIAN
    # neighbor similarity. A query in empty space has very few.
    median_nn = index.nn_percentiles.get(50, float(np.median(index.nn_sims)))
    density = int((sims >= median_nn).sum())

    # concentration: 1 - normalized entropy over the docs of the top-k.
    # High => support comes from one coherent source; low => scattered fragments.
    docs = [h.chunk.doc for h in hits]
    concentration = _doc_concentration(docs)

    sig = CoverageSignals(top1, score_gap, density, concentration)
    return RetrievalResult(query=query, hits=hits, signals=sig)


def _doc_concentration(docs: list[str]) -> float:
    if not docs:
        return 0.0
    _, counts = np.unique(docs, return_counts=True)
    p = counts / counts.sum()
    entropy = -(p * np.log(p + 1e-12)).sum()
    max_entropy = np.log(len(docs)) if len(docs) > 1 else 1.0
    return float(1.0 - entropy / max_entropy) if max_entropy > 0 else 1.0
