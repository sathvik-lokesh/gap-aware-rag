"""In-memory numpy vector store + the corpus self-statistics that make
gap detection possible.

KEY IDEA (leave-one-out pseudo-queries): a question never lands as close to
a passage as two passages land to each other, so we cannot calibrate against
doc<->doc similarity. Instead, at index time we re-embed every chunk *as a
query* and find its best match among the OTHER chunks. Each chunk's content
is, by definition, answerable from the corpus — so this distribution is
exactly 'what a top1 score looks like when the answer IS present.' THAT is
the yardstick the gap detector calibrates against at query time.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import INDEX_DIR


@dataclass
class Chunk:
    id: int
    text: str
    doc: str


@dataclass
class Index:
    vectors: np.ndarray          # (N, D) document vectors, L2-normalized
    chunks: list[Chunk]
    # Calibration: top1 score of each chunk used as a leave-one-out query.
    # This is the 'answerable query' distribution.
    nn_sims: np.ndarray = field(default=None)   # (N,)
    nn_percentiles: dict = field(default_factory=dict)  # percentile -> sim value

    # ---- calibration ----
    def calibrate(self, pseudo_query_vectors: np.ndarray) -> None:
        """pseudo_query_vectors are realistic short questions (LLM-generated,
        query-prefixed) whose answers live in the corpus. Each one's top1
        against the document vectors samples the answerable-query distribution.
        No diagonal exclusion: a real question SHOULD match its source chunk."""
        sim = pseudo_query_vectors @ self.vectors.T  # (M, N), pseudo-q vs docs
        self.nn_sims = sim.max(axis=1)               # best match per pseudo-query
        ps = [10, 25, 40, 50, 60, 75, 90]
        self.nn_percentiles = {
            p: float(np.percentile(self.nn_sims, p)) for p in ps
        }

    def sim_to_percentile(self, sim: float) -> float:
        """Where does a given similarity fall within the corpus NN distribution?
        Returns 0-100. Low => query is in sparse/empty embedding space."""
        if self.nn_sims is None:
            raise RuntimeError("call calibrate() first")
        return float((self.nn_sims < sim).mean() * 100.0)

    # ---- persistence ----
    def save(self, path: Path = INDEX_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        if self.nn_sims is not None:
            np.save(path / "nn_sims.npy", self.nn_sims)
        (path / "chunks.json").write_text(
            json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False)
        )
        (path / "percentiles.json").write_text(json.dumps(self.nn_percentiles))

    @classmethod
    def load(cls, path: Path = INDEX_DIR) -> "Index":
        vectors = np.load(path / "vectors.npy")
        chunks = [Chunk(**d) for d in json.loads((path / "chunks.json").read_text())]
        idx = cls(vectors=vectors, chunks=chunks)
        nn_path = path / "nn_sims.npy"
        if nn_path.exists():
            idx.nn_sims = np.load(nn_path)
            idx.nn_percentiles = json.loads((path / "percentiles.json").read_text())
        return idx
