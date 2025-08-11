from __future__ import annotations

"""HTML preprocessing for SEC filings.

Goals:
- Keep HTML structure for downstream parsers
- Drop scripts/styles/nav/footers
- Unwrap purely-presentational tags
- Remove repeated page artifacts (e.g., "Apple Inc. | 2024 Form 10-K | 21")
- Normalize whitespace entities
"""

import re
from bs4 import BeautifulSoup, Comment


PAGE_ARTIFACT_RE = re.compile(
    r"\b(?:Table of Contents|Index)\b|\bForm\s+10-[KQ]\b\s*\|\s*\d{4}\b|\u00a0",
    flags=re.IGNORECASE,
)


def sanitize_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove noisy containers and non-content
    for tag_name in ["script", "style", "svg", "noscript", "iframe", "nav", "footer", "img"]:
        for t in soup.find_all(tag_name):
            t.decompose()

    # Remove comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        c.extract()

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
    # Collapse excessive whitespace between tags/text
    html = re.sub(r"\s+", " ", html)
    return html.strip()


