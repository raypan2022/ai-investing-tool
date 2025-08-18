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


def query_index(ticker: str, query: str, *, top_k: int = 5, form: Optional[str] = None, bucket: Optional[str] = None) -> List[dict]:
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

    # --- Minimal gating and light penalties; heavy rerank happens later ---

    adjusted: List[float] = []
    anchors_mask: List[bool] = []
    for i, r in enumerate(recs):
        text = r.get("text", "")
        meta = r.get("meta", {}) or {}
        heading = meta.get("heading")
        lower_text = text.lower()

        # Numeric/anchor detection (hard preference): numbers or table anchors
        anchor = bool(PF_NUM_RE.search(text) or PF_TABLE_RE.search(text))
        anchors_mask.append(anchor)

        # Base similarity
        base = float(sims[i])

        # Anchored boost (tiny)
        ab = 0.05 if anchor else 0.0

        # Boilerplate penalty when no numbers
        bp = -0.40 if (not anchor and BOILERPLATE_RE.search(text)) else 0.0

        adjusted.append(base + ab + bp)

    # Hard gate preference: rank anchored first, then backfill if needed
    adjusted_np = np.array(adjusted)
    anchors_np = np.array(anchors_mask)
    anchor_indices = np.where(anchors_np)[0]
    non_anchor_indices = np.where(~anchors_np)[0]

    ranked_anchor = anchor_indices[np.argsort(adjusted_np[anchor_indices])[::-1]]
    selected = list(ranked_anchor[: max(1, top_k)])
    if len(selected) < max(1, top_k):
        ranked_non_anchor = non_anchor_indices[np.argsort(adjusted_np[non_anchor_indices])[::-1]]
        need = max(1, top_k) - len(selected)
        selected.extend(list(ranked_non_anchor[:need]))

    return [
        {
            "text": recs[i]["text"],
            "source": recs[i]["source"],
            "score": float(adjusted[i]),
            "meta": recs[i].get("meta", {}),
        }
        for i in selected
    ]


TOPIC_QUERIES = {
    "10-Q": {
        "business":    "Item 2 MD&A segment operating performance net sales by geography/segment 'dollars in millions' ASP units mix FX 'primarily due to' 'offset by' significant customers vendor concentration inventories DSO DIO DPO",
        "performance": "Item 2 Results gross margin percentage bps operating margin cost of sales opex drivers ASP shipments 'increase (decrease)' FX impact table",
        "liquidity":   "Item 2 Liquidity free cash flow capex share repurchase remaining authorization dividend per share restricted cash escrow debt maturities schedule covenants revolver interest commercial paper purchase obligations tax payable",
    },
    "10-K": {
        "business":    "Item 7 MD&A segment/geo net sales table 'dollars in millions' ASP units mix FX significant customers vendor concentration inventories DSO DIO DPO",
        "performance": "Item 7 MD&A margin bridge bps pricing mix volume FX outlook 'gross margin percentage' operating margin cost of sales opex table; Item 7A sensitivity table amounts 'hypothetical interest rate'",
        "liquidity":   "Item 7 Liquidity cash requirements FCF capex buybacks dividends remaining authorization debt maturity schedule interest facilities commercial paper purchase obligations tax payable",
    },
}



# --- Post-filter/rerank registry ---
# Shared regex for rerank
PF_NUM_RE = re.compile(r"(\$[0-9][0-9,\.]*|[0-9]+(?:\.[0-9]+)?%|\b[1-9][0-9]{1,4}\s?bps\b)", flags=re.IGNORECASE)
PF_TABLE_RE = re.compile(r"(dollars in millions|following\s+table|increase\s*\(\s*decrease\s*\)|maturity\s+schedule|sensitivity\s+table)", flags=re.IGNORECASE)
PF_CAUSE_RE = re.compile(r"(primarily due to|offset by|attributable to|drivers|impact)", flags=re.IGNORECASE)
PF_CURRENT_RE = re.compile(r"(as of\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}|for the\s+(three|nine)\s+months\s+ended)", flags=re.IGNORECASE)

# Generic boilerplate/methodology phrases used only for a light penalty when no numbers are present
BOILERPLATE_RE = re.compile(
    r"\b(may|could|might|uncertain|adverse impact|believes|is subject to|in accordance with|VAR\s+model|Monte\s+Carlo|segments?\s+.*managed\s+separately|we\s+evaluate\s+the\s+performance\s+of|trade\s+.*disputes)\b",
    flags=re.IGNORECASE,
)

ALLOW_SECTIONS = {
    "business": ("item 2", "item 7"),
    "performance": ("item 2", "item 7", "item 7a"),
    "liquidity": ("item 2", "item 7"),
}

NEG_PERF_RE = re.compile(r"(lease|right-of-use|ROU|VAR\s+model|Monte\s+Carlo|accounting\s+policy)", flags=re.IGNORECASE)
NEG_BUS_RE = re.compile(r"(segments?\s+.*managed\s+separately|we\s+evaluate\s+the\s+performance\s+of)", flags=re.IGNORECASE)
NEG_LIQ_RE = re.compile(r"(recent\s+accounting\s+pronouncements|\bASU\b|\bASC\b)", flags=re.IGNORECASE)
PERF_ROUTE_OUT_RE = re.compile(r"(inventories|inventory|vendor\s+non-trade)", flags=re.IGNORECASE)


def _apply_topic_rerank(topic: str, hits: List[dict], latest_q_date: Optional[str]) -> List[dict]:
    filtered: List[dict] = []
    for h in hits:
        txt: str = h.get("text", "")
        heading: str = ((h.get("meta", {}) or {}).get("heading") or "").lower()
        if topic in ALLOW_SECTIONS:
            if not any(tok in heading for tok in ALLOW_SECTIONS[topic]):
                # Drop wrong sections entirely
                continue

        nnums = len(PF_NUM_RE.findall(txt))
        s = 2.0 * nnums
        if PF_TABLE_RE.search(txt):
            s += 6.0
        if PF_CAUSE_RE.search(txt):
            s += 2.0
        if PF_CURRENT_RE.search(txt):
            s += 2.0

        # Prefer most recent 10-Q
        filed = _parse_date_from_source(h.get("source", "") or "")
        if latest_q_date and filed == latest_q_date:
            s += 3.0

        # Bucket-specific negatives only when light on numbers
        if topic == "performance" and nnums < 2 and NEG_PERF_RE.search(txt):
            s -= 10.0
        if topic == "business" and nnums < 2 and NEG_BUS_RE.search(txt):
            s -= 8.0
        if topic == "liquidity" and nnums < 2 and NEG_LIQ_RE.search(txt):
            s -= 6.0

        # Route certain terms out of performance
        if topic == "performance" and PERF_ROUTE_OUT_RE.search(txt):
            s -= 6.0

        # Attach rerank score to score field so downstream sort uses it
        base = float(h.get("score", 0.0))
        h["score"] = base + s
        filtered.append(h)

    return filtered if filtered else hits


def _make_topic_filter(topic: str):
    return lambda hits, latest_q: _apply_topic_rerank(topic, hits, latest_q)


TOPIC_POST_FILTERS = {
    "business": _make_topic_filter("business"),
    "performance": _make_topic_filter("performance"),
    "liquidity": _make_topic_filter("liquidity"),
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

    topics = ["business", "performance", "liquidity"]
    context: Dict[str, List[str]] = {k: [] for k in topics}
    sources: List[Dict[str, object]] = []

    def fetch(form: str, topic: str) -> List[dict]:
        query = TOPIC_QUERIES.get(form, {}).get(topic, "")
        hits = query_index(t, query, top_k=12, form=form, bucket=topic)
        prefer = q_date if form == "10-Q" else k_date
        if prefer:
            fhits = [h for h in hits if _parse_date_from_source(h.get("source", "")) == prefer]
            return fhits or hits
        return hits

    seen: set = set()
    for topic in topics:
        # prefer Q for recency, then K; rank by score
        merged = fetch("10-Q", topic) + fetch("10-K", topic)
        post_filter = TOPIC_POST_FILTERS.get(topic)
        if post_filter:
            merged = post_filter(merged, q_date)
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