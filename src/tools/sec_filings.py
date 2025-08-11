from __future__ import annotations

"""SEC filings fetcher (development-friendly skeleton).

Design goals:
- For development: read/save filings under data/documents/{ticker}/ as plain text or HTML/PDF files you place there.
- For production: replace `fetch_and_cache_filings` with an EDGAR client (e.g., sec-api, edgar-downloader) and keep the same return shape.

This module does NOT build embeddings or indices. It only resolves local file paths and (optionally) downloads raw filings.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional
import glob
import shutil
from src.tools.html_preprocess import sanitize_html

SEC_FOLDER = "sec-edgar-filings"
# Use sensible defaults so local runs work without a .env
DOCS_DIR = os.getenv("DOCS_DIR")
VENDOR_DIR = os.getenv("VENDOR_DIR")
SEC_CONTACT_NAME = os.getenv("SEC_CONTACT_NAME")
SEC_CONTACT_EMAIL = os.getenv("SEC_CONTACT_EMAIL")


@dataclass
class FilingPath:
    ticker: str
    path: str  # absolute or workspace-relative path
    doc_type: Optional[str] = None  # e.g., 10-K, 10-Q, 8-K
    as_of: Optional[str] = None     # ISO date if known


def ensure_ticker_dir(ticker: str) -> str:
    t = ticker.upper()
    dir_path = os.path.join(DOCS_DIR, t)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def ensure_form_dir(ticker: str, form: str) -> str:
    # Flattened layout: use only the ticker directory
    return ensure_ticker_dir(ticker)


def list_local_filings(ticker: str) -> List[FilingPath]:
    """List files under data/documents/{ticker}/file.html (flattened)."""
    tdir = ensure_ticker_dir(ticker)
    filings: List[FilingPath] = []
    if not os.path.isdir(tdir):
        return filings
    for name in sorted(os.listdir(tdir)):
        if name.lower().endswith((".html", ".htm")):
            # Extract FORM and YYYYMMDD from filename: TICKER.FORM[.YYYYMMDD].html
            date_in_name = None
            form_in_name = None
            m = re.match(r"^[A-Z]+\.([A-Z0-9]+)(?:\.(\d{8}))?\.htm(l)?$", name, flags=re.IGNORECASE)
            if m:
                form_in_name = m.group(1).upper()
                date_in_name = m.group(2)
            # Normalize common forms
            doc_form = form_in_name
            if doc_form == "10K":
                doc_form = "10-K"
            elif doc_form == "10Q":
                doc_form = "10-Q"
            filings.append(
                FilingPath(
                    ticker=ticker.upper(),
                    path=os.path.join(tdir, name),
                    doc_type=doc_form,
                    as_of=date_in_name,
                )
            )
    return filings


def _extract_accession_date(accession_dir: str) -> Optional[str]:
    """Extract YYYYMMDD from first line of full-submission.txt if present.

    Expected first line resembles:
      <SEC-DOCUMENT>0000320193-25-000073.txt : 20250801
    Returns the 8-digit date string or None if not found.
    """
    try:
        full_path = os.path.join(accession_dir, "full-submission.txt")
        if not os.path.isfile(full_path):
            return None
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        match = re.search(r":\s*(\d{8})\b", first_line)
        if match:
            return match.group(1)
        return None
    except Exception:
        return None


def fetch_and_cache_filings(
    ticker: str,
    *,
    doc_types: Optional[List[str]] = None,
    limit: int = 5,
) -> List[FilingPath]:
    safe_ticker = ticker.upper()
    os.makedirs(DOCS_DIR, exist_ok=True)

    target_types = [t.upper() for t in (doc_types or ["10-K", "10-Q"])]

    try:
        from sec_edgar_downloader import Downloader  # type: ignore
        dl: Optional[object] = Downloader(SEC_CONTACT_NAME, SEC_CONTACT_EMAIL, VENDOR_DIR)
    except Exception:
        dl = None

    if dl is not None:
        for form in target_types:
            try:
                # Always fetch human-readable detail docs; we only keep HTML artifacts downstream
                dl.get(form, safe_ticker, limit=limit, download_details=True)
            except Exception:
                continue

    # Copy the most useful files per filing into data/documents/{TICKER}/{FORM}/
    copied: List[FilingPath] = []
    for form in target_types:
        # Assume downloader layout: <vendor_root>/<TICKER>/<FORM>/<ACCESSION>/
        form_root = os.path.join(VENDOR_DIR, SEC_FOLDER, safe_ticker, form)
        if not os.path.isdir(form_root):
            continue
        # Each accession directory (e.g., 0000320193-24-000123)
        for acc_dir in sorted(glob.glob(os.path.join(form_root, "*"))):
            if not os.path.isdir(acc_dir):
                continue
            # Simple, non-recursive: pick primary-document*.html if present, else any *.html/*.htm in this folder
            primary = glob.glob(os.path.join(acc_dir, "primary-document*.html")) + glob.glob(os.path.join(acc_dir, "primary-document*.htm"))
            src = None
            
            if primary:
                src = primary[0]
            else:
                fallback = glob.glob(os.path.join(acc_dir, "*.html")) + glob.glob(os.path.join(acc_dir, "*.htm"))
                if not fallback:
                    continue
                src = fallback[0]

            # Flattened destination directory
            dest_dir = ensure_ticker_dir(safe_ticker)

            # Build destination file name: TICKER.FORM.YYYYMMDD.html (FORM without dashes)
            normalized_form = form.replace("-", "").upper()
            date_str = _extract_accession_date(acc_dir)
            if date_str:
                dest_filename = f"{safe_ticker}.{normalized_form}.{date_str}.html"
            else:
                dest_filename = f"{safe_ticker}.{normalized_form}.html"

            dest_path = os.path.join(dest_dir, dest_filename)
            # Replace any older files for this form
            normalized_form = form.replace("-", "").upper()
            for old in glob.glob(os.path.join(dest_dir, f"{safe_ticker}.{normalized_form}*.htm*")):
                try:
                    if os.path.abspath(old) != os.path.abspath(dest_path):
                        os.remove(old)
                except Exception:
                    pass
            # Write sanitized HTML
            try:
                with open(src, "r", encoding="utf-8", errors="ignore") as rf:
                    raw_html = rf.read()
                cleaned = sanitize_html(raw_html)
                with open(dest_path, "w", encoding="utf-8") as wf:
                    wf.write(cleaned)
            except Exception:
                # Fallback to raw copy if sanitization/write fails
                try:
                    shutil.copyfile(src, dest_path)
                except Exception:
                    continue
            copied.append(
                FilingPath(
                    ticker=safe_ticker,
                    path=dest_path,
                    doc_type=form,
                    as_of=date_str,
                )
            )

    return copied or list_local_filings(safe_ticker)


