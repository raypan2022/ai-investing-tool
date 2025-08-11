import os
import re
import pytest
from dotenv import load_dotenv

load_dotenv()


from src.tools.sec_filings import fetch_and_cache_filings, list_local_filings


def test_fetch_latest_10k_10q_live():
    tickers = ["AAPL", "NVDA"]
    for ticker in tickers:
        filings = fetch_and_cache_filings(ticker, doc_types=["10-K", "10-Q"], limit=1)
        assert isinstance(filings, list)
        assert len(filings) >= 1
        print(f"\nSEC filings for {ticker} -> data/documents/{ticker.upper()}/:")
        for f in filings:
            print(f" - {getattr(f, 'doc_type', None)} | {f.path}")
            # Path exists and is under data/documents/{TICKER}/ (flattened)
            assert os.path.exists(f.path)
            expected_dir = os.path.join("data", "documents", ticker.upper())
            print(f"   expected_dir: {expected_dir}")
            print(f"   path_norm:    {os.path.normpath(f.path)}")
            assert os.path.normpath(f.path).startswith(os.path.normpath(expected_dir))

            # Filename follows TICKER.FORM[.YYYYMMDD].html and as_of matches YYYYMMDD if present
            filename = os.path.basename(f.path)
            print(f"   filename:     {filename}")
            form_no_dash = f.doc_type.replace("-", "")
            m = re.match(rf"^{ticker.upper()}\.{form_no_dash}(?:\.(\d{{8}}))?\.html$", filename, flags=re.IGNORECASE)
            assert m is not None
            date_in_name = m.group(1)
            print(f"   parsed_date:  {date_in_name}")
            print(f"   as_of:        {getattr(f, 'as_of', None)}")
            if date_in_name:
                assert f.as_of == date_in_name


def test_list_local_filings_print():
    """Smoke test: listing local filings should not error; prints whatever is present."""
    ticker = os.getenv("TEST_SEC_TICKER", "AAPL")
    filings = list_local_filings(ticker)
    assert isinstance(filings, list)
    print(f"\nLocal filings for {ticker} (if any):")
    for f in filings:
        print(f" - {getattr(f, 'doc_type', None)} | {f.path}")
        # Validate directory structure (flattened)
        expected_dir = os.path.join("data", "documents", ticker.upper())
        print(f"   expected_dir: {expected_dir}")
        print(f"   path_norm:    {os.path.normpath(f.path)}")
        assert os.path.normpath(f.path).startswith(os.path.normpath(expected_dir))

        # Validate filename and as_of extraction
        filename = os.path.basename(f.path)
        print(f"   filename:     {filename}")
        form_no_dash = f.doc_type.replace("-", "")
        m = re.match(rf"^{ticker.upper()}\.{form_no_dash}(?:\.(\d{{8}}))?\.html$", filename, flags=re.IGNORECASE)
        assert m is not None
        date_in_name = m.group(1)
        print(f"   parsed_date:  {date_in_name}")
        print(f"   as_of:        {getattr(f, 'as_of', None)}")
        if date_in_name:
            assert f.as_of == date_in_name


