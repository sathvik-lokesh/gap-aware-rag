"""Central config. Keep all tunables here so experiments are reproducible."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
# Max chunks sampled to build the calibration distribution (one LLM pseudo-query
# each). Enough to estimate percentiles without an LLM call per corpus chunk.
CALIBRATION_SAMPLE = 60
EMBED_MODEL = "nomic-embed-text"   # 768-dim, fast, fully local
CHAT_MODEL = "qwen2.5:7b"          # planner / agent reasoning
FAST_MODEL = "qwen2.5:3b"          # cheap calls (judging, rewriting)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index_store"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    # Chunking. Smaller = each chunk holds ~one idea, so a specific question
    # isn't diluted by unrelated sentences in the same chunk. This single knob
    # moves retrieval precision more than almost anything else.
    chunk_chars: int = 320
    chunk_overlap: int = 60


@dataclass
class GapConfig:
    """Thresholds are expressed as PERCENTILES of the corpus's own
    nearest-neighbor similarity distribution, NOT raw cosine values.
    That is the whole point: 'good' is defined relative to THIS corpus."""
    # If the query's best match is below this percentile of in-corpus
    # neighbor similarities, the query is landing in sparse/empty space.
    gap_percentile: float = 25.0
    answerable_percentile: float = 60.0
    # Density radius is the corpus median NN similarity; a query needs at
    # least this many neighbors above it to count as well-supported.
    min_dense_neighbors: int = 3
    # Absolute cosine floor: if even the best chunk is below this, retrieval
    # returned nothing topically related, so we abstain WITHOUT spending an LLM
    # call. Above it, the LLM verifier is the arbiter (the GAP verdict is only a
    # prior, never a hard gate — that is what lets verification rescue the
    # 'CEO present but low-scoring' case).
    hard_abstain_floor: float = 0.45


CFG = RetrievalConfig()
GAP = GapConfig()
