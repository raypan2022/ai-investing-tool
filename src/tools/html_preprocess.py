from __future__ import annotations

"""HTML preprocessing for SEC filings.

Goals:
- Keep HTML structure for downstream parsers
- Drop scripts/styles/nav/footers
- Unwrap purely-presentational tags
- Remove repeated page artifacts (e.g., "Apple Inc. | 2024 Form 10-K | 21")
- Normalize whitespace entities and Unicode
"""

import re
import unicodedata
from bs4 import BeautifulSoup, Comment


PAGE_ARTIFACT_RE = re.compile(
    (
        r"(?:\bTable of Contents\b|\bIndex\b|"
        r"Page\s+\d+\s+of\s+\d+|"
        r"UNITED STATES SECURITIES AND EXCHANGE COMMISSION|"
        r"Washington,\s*D\.C\.\s*20549|"
        r"Cover\s*Page|"
        r"\u00a0)"
    ),
    flags=re.IGNORECASE,
)

# Map common smart punctuation to ASCII so downstream regexes match cleanly
SMART_PUNCT_MAP = {
    "\u2018": "'",  # left single
    "\u2019": "'",  # right single
    "\u201c": '"',  # left double
    "\u201d": '"',  # right double
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus
    "\u2022": "-",  # bullet
}


def _normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    for src, dst in SMART_PUNCT_MAP.items():
        normalized = normalized.replace(src, dst)
    return normalized


def sanitize_html(raw_html: str) -> str:
    # Normalize encoding and smart punctuation before parsing
    raw_html = _normalize_unicode(raw_html)
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove noisy containers and non-content
    for tag_name in [
        "script",
        "style",
        "svg",
        "noscript",
        "iframe",
        "nav",
        "footer",
        "img",
        "form",
        "header",
        "aside",
    ]:
        for t in soup.find_all(tag_name):
            t.decompose()

    # Remove comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        c.extract()

    # Handle inline iXBRL: unwrap visible facts, drop hidden ones
    for t in list(soup.find_all(True)):
        name = (getattr(t, "name", "") or "").lower()
        if name.startswith("ix:"):
            hidden_attr = (t.get("hidden") or "").strip().lower()
            style = (t.get("style") or "").replace(" ", "").lower()
            is_hidden = hidden_attr in {"true", "1"} or "display:none" in style
            if is_hidden:
                t.decompose()
            else:
                t.unwrap()

    # Drop XBRL/link-only containers that add noise
    for xbrl_name in ["link:linkbase", "link:roleRef", "xbrli:context", "xbrli:unit"]:
        for t in soup.find_all(xbrl_name):
            t.decompose()

    # Unwrap some purely-presentational tags to reduce nesting
    # Keep <span> because many filings wrap headings in spans; we'll strip attrs later
    for t in soup.find_all(["font", "u"]):
        t.unwrap()

    # Whitelist of structural tags to keep; unwrap others but keep their text
    allowed = {
        "html", "body",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "div", "br",
        "ul", "ol", "li",
        "table", "thead", "tbody", "tr", "th", "td",
        "em", "strong", "b", "i",
        "span", "a",
    }
    for t in list(soup.find_all(True)):
        if t.name not in allowed:
            t.unwrap()

    # Strip attributes (especially inline styles); allow minimal table attrs
    allowed_attrs = {
        "th": {"colspan", "rowspan", "scope"},
        "td": {"colspan", "rowspan", "headers"},
        "table": set(),
        "tr": set(),
        "thead": set(),
        "tbody": set(),
        # Preserve IDs on headings/spans so section anchors survive
        "h1": {"id"},
        "h2": {"id"},
        "h3": {"id"},
        "h4": {"id"},
        "h5": {"id"},
        "h6": {"id"},
        "span": {"id"},
        # We'll special-handle <a> tags below
        "a": set(),
    }
    for t in soup.find_all(True):
        keep = allowed_attrs.get(t.name, set())
        for attr in list(t.attrs.keys()):
            if attr not in keep:
                del t.attrs[attr]

    # Remove link-heavy/XBRL anchors and unwrap benign anchors
    for a in list(soup.find_all("a")):
        txt = a.get_text(" ", strip=True) or ""
        low = txt.lower()
        if "http://" in low or "https://" in low or "fasb.org" in low or "us-gaap" in low or "xbrl" in low:
            a.decompose()
        else:
            a.unwrap()

    # Remove trivial empty blocks after unwrapping/stripping
    for t in soup.find_all(["p", "div", "li", "th", "td"]):
        if not t.get_text(strip=True):
            t.decompose()

    # Serialize
    html = str(soup)

    # Normalize nbsp to space and remove common page artifacts
    html = html.replace("\u00a0", " ")
    html = PAGE_ARTIFACT_RE.sub(" ", html)

    # Drop residual URL-like tokens (e.g., taxonomy links rendered as text)
    html = re.sub(r"https?://\S+", " ", html)
    # Preserve paragraph structure by not collapsing all whitespace globally
    return html.strip()


