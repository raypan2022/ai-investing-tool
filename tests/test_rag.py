import os
import json
import pytest
from dotenv import load_dotenv

load_dotenv()

from src.tools.sec_filings import list_local_filings
from src.tools.rag import build_index_for_ticker, query_index, INDEX_DIR, EMBED_MODEL_NAME
from src.tools.filings_processing import chunks_out_path_for_filing, read_chunks_jsonl


@pytest.mark.order(1)
def test_build_index_from_local_filings():
    ticker = os.getenv("TEST_SEC_TICKER", "AAPL")
    filings = list_local_filings(ticker)
    assert filings, "Expected at least one local filing to exist for tests"

    # Ensure chunks path resolution and creation during index build
    first_path = filings[0].path
    chunks_path = chunks_out_path_for_filing(ticker, first_path)
    print(f"\nRAG build test for {ticker}")
    print(f" - first filing: {first_path}")
    print(f" - chunks path: {chunks_path}")

    index_path = build_index_for_ticker(ticker, filing_paths=[f.path for f in filings])
    print(f" - index path:  {index_path}")

    assert os.path.isfile(chunks_path), "Chunks JSONL should be created for the first filing"
    assert os.path.isfile(index_path), "Index JSON should be created for the ticker"

    # Validate chunks shape
    chunks = read_chunks_jsonl(chunks_path)
    print(f" - num chunks:  {len(chunks)}")
    assert len(chunks) > 0
    c0 = chunks[0]
    assert c0.text and c0.heading and c0.section_id and c0.source

    # Validate index structure
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("model") == EMBED_MODEL_NAME
    records = data.get("records", [])
    print(f" - num records: {len(records)}")
    assert records and "embedding" in records[0]


@pytest.mark.order(2)
def test_query_index_returns_chunks():
    ticker = os.getenv("TEST_SEC_TICKER", "AAPL")
    index_path = os.path.join(INDEX_DIR, f"{ticker.upper()}_mini.json")
    assert os.path.isfile(index_path), "Index must exist from previous test"

    results = query_index(ticker, query=f"what are {ticker}'s revenue and outlook and trends?")
    print(f"\nQuery results for {ticker} (top {len(results)}):")
    for r in results:
        print(f" - score={r.get('score'):.4f} | source={os.path.basename(r.get('source',''))}")
        meta = r.get("meta", {})
        if meta:
            print(f"   meta: {meta}")

    assert isinstance(results, list)
    assert len(results) > 0
    r0 = results[0]
    assert "text" in r0 and "source" in r0 and "score" in r0

