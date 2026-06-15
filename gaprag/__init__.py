"""Gap-Aware Agentic RAG — a RAG system that models the boundary of its own
knowledge: it detects when a question lands outside the corpus and abstains
(naming the gap) instead of hallucinating."""
from .index import Index, Chunk
from .ingest import build_index
from .retriever import retrieve, RetrievalResult, CoverageSignals
from .gapdetector import assess, Verdict, GapAssessment
from .agent import run, AgentResult, SubAnswer, decompose, verify_fact

__all__ = [
    "Index", "Chunk", "build_index", "retrieve", "RetrievalResult",
    "CoverageSignals", "assess", "Verdict", "GapAssessment",
    "run", "AgentResult", "SubAnswer", "decompose", "verify_fact",
]
