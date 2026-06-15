"""Build + calibrate the index over the real SQuAD corpus.

Run:  uv run python scripts/build_index_squad.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaprag.ingest import build_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus_squad"
INDEX = ROOT / "index_store_squad"


def main():
    t0 = time.time()
    print(f"building index from {CORPUS} ...")
    idx = build_index(CORPUS)
    idx.save(INDEX)
    print(f"chunks={len(idx.chunks)}  dim={idx.vectors.shape[1]}  "
          f"saved -> {INDEX}")
    print("calibration percentiles:",
          {p: round(v, 3) for p, v in idx.nn_percentiles.items()})
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
