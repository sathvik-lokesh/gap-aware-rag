"""Read documents, chunk them, embed, and build a calibrated Index."""
from __future__ import annotations
from pathlib import Path

from .config import CFG, DATA_DIR
from .embeddings import embed_documents
from .calibrate import answerable_query_vectors
from .index import Index, Chunk


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Character-window chunking with overlap. Simple on purpose — you can
    swap in sentence/semantic chunking later and MEASURE the difference."""
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def build_index(data_dir: Path = DATA_DIR) -> Index:
    chunks: list[Chunk] = []
    for fp in sorted(data_dir.glob("**/*")):
        if fp.suffix.lower() not in {".txt", ".md"}:
            continue
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        for piece in chunk_text(raw, CFG.chunk_chars, CFG.chunk_overlap):
            chunks.append(Chunk(id=len(chunks), text=piece, doc=fp.name))

    if not chunks:
        raise RuntimeError(f"No .txt/.md files found in {data_dir}")

    texts = [c.text for c in chunks]
    doc_vectors = embed_documents(texts)        # stored, searched against
    idx = Index(vectors=doc_vectors, chunks=chunks)
    # Calibrate against realistic LLM-generated pseudo-queries.
    idx.calibrate(answerable_query_vectors(chunks))
    return idx
