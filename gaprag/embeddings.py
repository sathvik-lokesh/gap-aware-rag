"""Ollama embedding client. Raw HTTP on purpose so you SEE the call.

CRITICAL nomic-embed-text detail: this model is trained with task prefixes.
Documents must be embedded as 'search_document: <text>' and queries as
'search_query: <text>'. Omit them and query<->document similarity collapses.
This is the #1 silent footgun with this embedder.
"""
from __future__ import annotations
import numpy as np
import requests

from .config import OLLAMA_URL, EMBED_MODEL

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _embed_raw(text: str, model: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return _normalize(np.asarray(r.json()["embedding"], dtype=np.float32))


def embed_document(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    return _embed_raw(DOC_PREFIX + text, model)


def embed_query(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    return _embed_raw(QUERY_PREFIX + text, model)


def embed_documents(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    vecs = [embed_document(t, model) for t in texts]
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)


def embed_queries(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    vecs = [embed_query(t, model) for t in texts]
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
