from __future__ import annotations

"""RAG utilities (development-friendly skeleton).

Stages
- Ingestion (dev): parse curated HTML → sections → chunks → embeddings → JSON index on disk
- Query: load index for ticker → retrieve top-k

Future
- Swap JSON on disk with FAISS or a vector DB without changing interfaces.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Iterable, Tuple

from src.tools.filings_processing import (
    FilingChunk,
    read_chunks_jsonl,
    parse_html_to_sections,
    chunk_sections,
    chunks_out_path_for_filing,
    write_chunks_jsonl,
)


INDEX_DIR = os.path.join("data", "indices")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # dev default


@dataclass
class RagChunk:
    text: str
    source: str  # path or doc id
    meta: dict


def read_texts_from_paths(paths: List[str]) -> List[Tuple[str, str]]:
    """Return list of (text, path). Non-text like PDF should be pre-converted upstream for dev simplicity."""
    out: List[Tuple[str, str]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                out.append((f.read(), p))
        except Exception:
            continue
    return out


def simple_chunk(text: str, *, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + max_chars])
        i += max(1, max_chars - overlap)
    return chunks


def build_index_for_ticker(ticker: str, *, filing_paths: List[str]) -> str:
    """Build a naive JSON index from section-aware chunks.

    Steps (dev): ensure chunks exist → compute embeddings → write JSON index.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    from sentence_transformers import SentenceTransformer

    # Ensure chunks exist for each filing
    all_chunks: List[FilingChunk] = []
    for html_path in filing_paths:
        out_chunks_path = chunks_out_path_for_filing(ticker, html_path)
        if not os.path.isfile(out_chunks_path):
            sections = parse_html_to_sections(html_path)
            chunks = chunk_sections(sections)
            write_chunks_jsonl(chunks, out_chunks_path)
        all_chunks.extend(read_chunks_jsonl(out_chunks_path))

    if not all_chunks:
        # Fallback: read raw text, simple chunk
        texts: List[RagChunk] = []
        for text, path in read_texts_from_paths(filing_paths):
            for idx, chunk in enumerate(simple_chunk(text)):
                texts.append(RagChunk(text=chunk, source=path, meta={"chunk": idx}))
    else:
        texts = [RagChunk(text=c.text, source=c.source, meta={"chunk_id": c.chunk_id, "heading": c.heading, "section_id": c.section_id, "offsets": c.offsets}) for c in all_chunks]

    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode([c.text for c in texts], convert_to_numpy=True, normalize_embeddings=True)

    records: List[dict] = []
    for c, vec in zip(texts, embeddings):
        records.append({
            "text": c.text,
            "source": c.source,
            "meta": c.meta,
            "embedding": vec.tolist(),
        })

    index_path = os.path.join(INDEX_DIR, f"{ticker.upper()}_mini.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"model": EMBED_MODEL_NAME, "records": records}, f)
    return index_path


def query_index(ticker: str, query: str, *, top_k: int = 5) -> List[dict]:
    """Dev-only: cosine similarity over the saved JSON embeddings for a ticker."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    index_path = os.path.join(INDEX_DIR, f"{ticker.upper()}_mini.json")
    if not os.path.isfile(index_path):
        return []

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])
    if not records:
        return []

    model = SentenceTransformer(EMBED_MODEL_NAME)
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    embs = np.array([r["embedding"] for r in records])
    sims = (embs @ qvec)
    top_idx = sims.argsort()[::-1][: top_k]

    return [
        {
            "text": records[i]["text"],
            "source": records[i]["source"],
            "score": float(sims[i]),
            "meta": records[i].get("meta", {}),
        }
        for i in top_idx
    ]


