from __future__ import annotations

"""
Minimal SEC RAG utilities focused on high-signal extraction.

Provides:
- build_indices_for_ticker_by_form: build tiny JSON indices per form (10-Q/10-K)
- query_index: cosine search over saved embeddings
- build_llm_context: compact, data-rich snippets for LLM prompts
"""

import os
import re
import json
from functools import lru_cache
from typing import List, Tuple, Optional, Dict

from src.tools.filings_processing import (
    FilingChunk,
    read_chunks_jsonl,
    parse_html_to_sections,
    chunk_sections,
    chunks_out_path_for_filing,
    write_chunks_jsonl,
)
from src.tools.sec_filings import list_local_filings


INDEX_DIR = os.path.join("data", "indices")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def _infer_form_from_filename(path: str) -> Optional[str]:
    b = os.path.basename(path).upper()
    if ".10K." in b:
        return "10-K"
    if ".10Q." in b:
        return "10-Q"
    return None


def _index_path_for(ticker: str, form: Optional[str]) -> str:
    tdir = os.path.join(INDEX_DIR, ticker.upper())
    os.makedirs(tdir, exist_ok=True)
    if form:
        return os.path.join(tdir, f"{ticker.upper()}_{form.replace('-', '')}_mini.json")
    return os.path.join(tdir, f"{ticker.upper()}_mini.json")


def _prepare_chunks(ticker: str, filing_paths: List[str]) -> List[FilingChunk]:
    out: List[FilingChunk] = []
    for html_path in filing_paths:
        cpath = chunks_out_path_for_filing(ticker, html_path)
        if not os.path.isfile(cpath):
            sections = parse_html_to_sections(html_path)
            chunks = chunk_sections(sections)
            write_chunks_jsonl(chunks, cpath)
        out.extend(read_chunks_jsonl(cpath))
    return out


def _read_texts(paths: List[str]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                items.append((f.read(), p))
        except Exception:
            pass
    return items


def _simple_chunk(text: str, *, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + max_chars])
        i += max(1, max_chars - overlap)
    return chunks


def build_index_for_ticker_form(ticker: str, *, filing_paths: List[str], form: str) -> str:
    os.makedirs(INDEX_DIR, exist_ok=True)

    filtered = [p for p in filing_paths if _infer_form_from_filename(p) == form]
    ipath = _index_path_for(ticker, form)
    if not filtered:
        with open(ipath, "w", encoding="utf-8") as f:
            json.dump({"model": EMBED_MODEL_NAME, "records": []}, f)
        return ipath

    chunks = _prepare_chunks(ticker, filtered)
    if not chunks:
        # Fallback to raw text if parsing failed
        payload: List[Dict[str, object]] = []
        for text, src in _read_texts(filtered):
            for i, part in enumerate(_simple_chunk(text)):
                payload.append({"text": part, "source": src, "meta": {"chunk_id": f"raw-{i}"}})
    else:
        payload = [
            {
                "text": c.text,
                "source": c.source,
                "meta": {
                    "chunk_id": c.chunk_id,
                    "heading": c.heading,
                    "section_id": c.section_id,
                    "offsets": c.offsets,
                },
            }
            for c in chunks
        ]

    model = get_embed_model()
    embeddings = model.encode([p["text"] for p in payload], convert_to_numpy=True, normalize_embeddings=True)
    for obj, vec in zip(payload, embeddings):
        obj["embedding"] = vec.tolist()

    with open(ipath, "w", encoding="utf-8") as f:
        json.dump({"model": EMBED_MODEL_NAME, "records": payload}, f)
    return ipath


def build_indices_for_ticker_by_form(ticker: str, *, filing_paths: List[str]) -> Dict[str, str]:
    return {form: build_index_for_ticker_form(ticker, filing_paths=filing_paths, form=form) for form in ("10-K", "10-Q")}


def query_index(ticker: str, query: str, *, top_k: int = 5, form: Optional[str] = None) -> List[dict]:
    import numpy as np

    ipath = _index_path_for(ticker, form)
    if not os.path.isfile(ipath):
        return []
    with open(ipath, "r", encoding="utf-8") as f:
        data = json.load(f)
    recs = data.get("records", [])
    if not recs:
        return []

    model = get_embed_model()
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    embs = np.array([r["embedding"] for r in recs])
    sims = (embs @ qvec)
    idx = sims.argsort()[::-1][: max(1, top_k)]
    return [
        {"text": recs[i]["text"], "source": recs[i]["source"], "score": float(sims[i]), "meta": recs[i].get("meta", {})}
        for i in idx
    ]


# --- Compact topic queries (signal > narrative) ---
TOPIC_QUERIES = {
    "10-Q": {
        "business":   "Item 2 MD&A Item 2 Results of Operations overview demand pricing ASP mix volume orders backlog channel inventory supply constraints known trends guidance 'primarily due to' 'offset by'",
        "performance":"Item 2 MD&A Item 2 Results of Operations revenue gross margin operating margin bps bridge pricing mix cost freight warranty utilization opex leverage by segment by geography FX ASP units",
        "liquidity":  "Item 2 Liquidity and Capital Resources operating cash flow free cash flow capex DSO DIO DPO working capital inventory reserves debt maturities covenants revolving credit availability interest expense variable fixed",
        "risks":      "Item 4 Controls and Procedures change in internal control disclosure controls ICFR material weakness remediation Item 1 Legal Proceedings new material investigation subpoena settlement cybersecurity incident going concern covenant breach waiver impairment",
    },
    "10-K": {
        "business":   "Item 7 MD&A overview results drivers pricing mix volume FX known trends guidance 'primarily due to' 'offset by' (avoid Item 1 Business)",
        "performance":"Item 7 MD&A Results of Operations revenue gross margin operating margin bps bridge pricing mix cost freight warranty utilization opex leverage by segment by geography FX ASP units yoy",
        "liquidity":  "Item 7 Liquidity and Capital Resources cash flows cash requirements free cash flow capex dividends buybacks ASR M&A debt maturities covenants interest rate sensitivity liquidity facilities",
        "risks":      "Item 9A Controls and Procedures ICFR material weakness remediation auditor opinion Item 3 Legal Proceedings new material investigation subpoena settlement cybersecurity incident going concern covenant breach waiver impairment",
    },
}



def _parse_date_from_source(path: str) -> Optional[str]:
    m = re.search(r"\.(\d{8})\.html$", os.path.basename(path), flags=re.IGNORECASE)
    return m.group(1) if m else None


def _form_from_source(path: str) -> Optional[str]:
    up = os.path.basename(path).upper()
    if ".10Q." in up:
        return "10-Q"
    if ".10K." in up:
        return "10-K"
    return None


def _latest_date_for_form(ticker: str, form: str) -> Optional[str]:
    filings = list_local_filings(ticker)
    dates = [f.as_of for f in filings if (f.doc_type or "").upper() == form.upper() and f.as_of]
    return max(dates) if dates else None


def _trim(text: str, max_chars: int = 650) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars]
    last_dot = max(cut.rfind("."), cut.rfind(";"))
    return cut[: last_dot + 1] if last_dot > int(max_chars * 0.6) else cut.rstrip() + "…"


def build_llm_context(ticker: str, max_per_topic: int = 4) -> Dict[str, object]:
    t = ticker.upper()
    q_date = _latest_date_for_form(t, "10-Q")
    k_date = _latest_date_for_form(t, "10-K")
    warnings: List[str] = []
    if not q_date:
        warnings.append("Latest 10-Q not found")
    if not k_date:
        warnings.append("Latest 10-K not found")

    topics = ["business", "performance", "liquidity", "risks"]
    context: Dict[str, List[str]] = {k: [] for k in topics}
    sources: List[Dict[str, object]] = []

    def fetch(form: str, topic: str) -> List[dict]:
        query = TOPIC_QUERIES.get(form, {}).get(topic, "")
        hits = query_index(t, query, top_k=12, form=form)
        prefer = q_date if form == "10-Q" else k_date
        if prefer:
            fhits = [h for h in hits if _parse_date_from_source(h.get("source", "")) == prefer]
            return fhits or hits
        return hits

    seen: set = set()
    for topic in topics:
        # prefer Q for recency, then K; rank by score
        merged = fetch("10-Q", topic) + fetch("10-K", topic)
        picked: List[dict] = []
        for h in sorted(merged, key=lambda r: float(r.get("score", 0.0)), reverse=True):
            cid = (h.get("meta", {}) or {}).get("chunk_id")
            if cid and cid in seen:
                continue
            picked.append(h)
            if cid:
                seen.add(cid)
            if len(picked) >= max_per_topic:
                break

        for h in picked:
            text = _trim(h.get("text", ""))
            context[topic].append(text)
            src = h.get("source", "")
            sources.append({
                "chunk_id": (h.get("meta", {}) or {}).get("chunk_id"),
                "form": _form_from_source(src),
                "filed_date": _parse_date_from_source(src),
                "section_title": (h.get("meta", {}) or {}).get("heading"),
                "source_path": src,
            })

    return {"ticker": t, "context": context, "sources": sources, "warnings": warnings}