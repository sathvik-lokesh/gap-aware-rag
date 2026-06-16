"""Central config. Keep all tunables here so experiments are reproducible."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"

# LLM backend: "ollama" (fully local, default) or "groq" (fast hosted, needs
# GROQ_API_KEY). Set GAPRAG_LLM_BACKEND=groq to route chat() through Groq's
# OpenAI-compatible endpoint — the eval measures the gap-aware ARCHITECTURE, not
# the model, so swapping in a faster/stronger verifier is sound (and far faster
# than 7b on this CPU box). The local path stays the reproducible default.
LLM_BACKEND = os.environ.get("GAPRAG_LLM_BACKEND", "ollama").lower()
GROQ_URL = "https://api.groq.com/openai/v1"
# Map the local Ollama model names the code already passes around to their Groq
# equivalents, so nothing else in the codebase has to change.
GROQ_MODEL_MAP = {
    "qwen2.5:7b": "llama-3.3-70b-versatile",  # verification / planning
    "qwen2.5:3b": "llama-3.1-8b-instant",     # cheap judging / rewriting
}
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
