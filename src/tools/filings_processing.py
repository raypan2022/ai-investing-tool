from __future__ import annotations

"""SEC filings processing utilities.

Functions to parse curated HTML into logical sections and then produce
section-aware text chunks ready for embedding.

Artifacts (dev):
- Chunks JSONL per filing under data/chunks/{TICKER}/{BASENAME}.jsonl
  where BASENAME is the HTML filename without extension (e.g., AAPL.10K.20250801)
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup


CHUNKS_ROOT = os.path.join("data", "chunks")


@dataclass
class FilingSection:
    section_id: str
    heading: str
    text: str
    source: str


@dataclass
class FilingChunk:
    chunk_id: str
    section_id: str
    heading: str
    text: str
    source: str
    offsets: Dict[str, int]


def _strip_unwanted_tags(soup: BeautifulSoup) -> None:
    for tag_name in ["script", "style", "nav", "footer", "noscript", "svg"]:
        for t in soup.find_all(tag_name):
            t.decompose()


def _is_heading_element(tag) -> bool:
    if tag.name and tag.name.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return True
    # Heuristic: SEC forms often use strong/b tags for section titles like "Item 7. Management's Discussion...".
    if tag.name and tag.name.lower() in {"b", "strong"}:
        text = tag.get_text(" ", strip=True)
        return bool(re.match(r"^item\s+\d+\.?", text, flags=re.IGNORECASE))
    return False


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_html_to_sections(path: str) -> List[FilingSection]:
    """Parse curated primary HTML into logical sections.

    - Drops script/style/nav/footer
    - Uses h1–h6 and Item X patterns as headings
    - Collapses whitespace
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except Exception:
        return []

    soup = BeautifulSoup(html, "lxml")
    _strip_unwanted_tags(soup)

    body = soup.body or soup
    current_heading = "Document"
    current_text_parts: List[str] = []
    sections: List[FilingSection] = []
    section_index = 0

    def flush_section():
        nonlocal section_index, current_heading, current_text_parts
        text = _normalize_whitespace("\n".join([t for t in current_text_parts if t.strip()]))
        if text:
            section_id = f"sec-{section_index:04d}"
            sections.append(FilingSection(section_id=section_id, heading=current_heading, text=text, source=path))
            section_index += 1
        current_text_parts = []

    for el in body.descendants:
        if getattr(el, "name", None) is None:
            continue
        if _is_heading_element(el):
            # Flush previous section
            flush_section()
            current_heading = _normalize_whitespace(el.get_text(" ", strip=True)) or current_heading
            continue
        if el.name and el.name.lower() in {"p", "div", "span", "li"}:
            txt = _normalize_whitespace(el.get_text(" ", strip=True))
            if txt:
                current_text_parts.append(txt)

    # Flush tail
    flush_section()
    return sections


def chunk_sections(
    sections: List[FilingSection], *, max_chars: int = 1200, overlap: int = 200
) -> List[FilingChunk]:
    chunks: List[FilingChunk] = []
    for s in sections:
        text = s.text
        if not text:
            continue
        start = 0
        part_index = 0
        while start < len(text):
            end = min(len(text), start + max_chars)
            chunk_text = text[start:end]
            chunk_id = f"{s.section_id}-p{part_index:03d}"
            chunks.append(
                FilingChunk(
                    chunk_id=chunk_id,
                    section_id=s.section_id,
                    heading=s.heading,
                    text=chunk_text,
                    source=s.source,
                    offsets={"start": start, "end": end},
                )
            )
            if end >= len(text):
                break
            start = max(0, end - overlap)
            part_index += 1
    return chunks


def chunks_out_path_for_filing(ticker: str, filing_html_path: str) -> str:
    base = os.path.splitext(os.path.basename(filing_html_path))[0]
    out_dir = os.path.join(CHUNKS_ROOT, ticker.upper())
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}.jsonl")


def write_chunks_jsonl(chunks: List[FilingChunk], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "section_id": c.section_id,
                        "heading": c.heading,
                        "text": c.text,
                        "source": c.source,
                        "offsets": c.offsets,
                    }
                )
                + "\n"
            )


def read_chunks_jsonl(path: str) -> List[FilingChunk]:
    items: List[FilingChunk] = []
    if not os.path.isfile(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                items.append(
                    FilingChunk(
                        chunk_id=obj["chunk_id"],
                        section_id=obj.get("section_id", ""),
                        heading=obj.get("heading", ""),
                        text=obj.get("text", ""),
                        source=obj.get("source", ""),
                        offsets=obj.get("offsets", {}),
                    )
                )
            except Exception:
                continue
    return items


