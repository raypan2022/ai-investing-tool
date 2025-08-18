import os
import json
import pytest
from dotenv import load_dotenv

load_dotenv()

from src.tools.sec_filings import list_local_filings, fetch_and_cache_filings
from src.tools.rag import build_indices_for_ticker_by_form, query_index, build_llm_context, INDEX_DIR, EMBED_MODEL_NAME
from src.tools.filings_processing import chunks_out_path_for_filing, read_chunks_jsonl


def test_build_index_from_local_filings():
    ticker = os.getenv("TEST_SEC_TICKER", "AAPL")
    # Ensure flattened docs exist
    fetch_and_cache_filings(ticker, doc_types=["10-K", "10-Q"], limit=1)
    filings = list_local_filings(ticker)
    assert filings, "Expected at least one local filing to exist for tests"

    # Ensure chunks path resolution and creation during index build
    first_path = filings[0].path
    chunks_path = chunks_out_path_for_filing(ticker, first_path)
    print(f"\nRAG build test for {ticker}")
    print(f" - first filing: {first_path}")
    print(f" - chunks path: {chunks_path}")

    idx_paths = build_indices_for_ticker_by_form(ticker, filing_paths=[f.path for f in filings])
    print(f" - index path (10-Q):  {idx_paths.get('10-Q')}")

    assert os.path.isfile(chunks_path), "Chunks JSONL should be created for the first filing"
    # Expect at least one of the indices to exist
    assert any(os.path.isfile(p) for p in idx_paths.values()), "At least one form index should be created for the ticker"

    # Validate chunks shape
    chunks = read_chunks_jsonl(chunks_path)
    print(f" - num chunks:  {len(chunks)}")
    assert len(chunks) > 0
    c0 = chunks[0]
    assert c0.text and c0.heading and c0.section_id and c0.source

    # Validate index structure (check whichever exists)
    for p in idx_paths.values():
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data.get("model") == EMBED_MODEL_NAME
            records = data.get("records", [])
            print(f" - num records in {os.path.basename(p)}: {len(records)}")
            assert isinstance(records, list)
            break


def test_query_index_returns_chunks():
    ticker = os.getenv("TEST_SEC_TICKER", "AAPL")
    index_q = os.path.join(INDEX_DIR, ticker.upper(), f"{ticker.upper()}_10Q_mini.json")
    index_k = os.path.join(INDEX_DIR, ticker.upper(), f"{ticker.upper()}_10K_mini.json")
    assert os.path.isfile(index_q) or os.path.isfile(index_k), "At least one index must exist from previous test"

    results = query_index(ticker, query=f"what are {ticker}'s revenue and outlook and trends?", form="10-K") + query_index(ticker, query=f"what are {ticker}'s revenue and outlook and trends?", form="10-Q")
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


def test_build_llm_context_packet(tmp_path):
    ticker = os.getenv("TEST_SEC_TICKER", "AAPL")
    # Ensure indices exist from prior tests; if not, trigger build
    index_q = os.path.join(INDEX_DIR, ticker.upper(), f"{ticker.upper()}_10Q_mini.json")
    index_k = os.path.join(INDEX_DIR, ticker.upper(), f"{ticker.upper()}_10K_mini.json")
    if not (os.path.isfile(index_q) or os.path.isfile(index_k)):
        fetch_and_cache_filings(ticker, doc_types=["10-K", "10-Q"], limit=1)
        filings = list_local_filings(ticker)
        build_indices_for_ticker_by_form(ticker, filing_paths=[f.path for f in filings])

    packet = build_llm_context(ticker, max_per_topic=3)
    ctx = packet.get("context", {})
    srcs = packet.get("sources", [])

    # Write full context and sources to a temp file for inspection
    out_path = tmp_path / f"{ticker.upper()}_llm_context.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Ticker: {ticker.upper()}\nWarnings: {packet.get('warnings')}\n\n")
        for topic in ["business", "performance", "liquidity"]:
            snippets = ctx.get(topic, [])
            f.write(f"[{topic}] ({len(snippets)})\n")
            for i, s in enumerate(snippets, 1):
                f.write(f"\n--- {topic} snippet {i} (len={len(s)}) ---\n")
                f.write(s)
                f.write("\n")
            f.write("\n")
        f.write("\nSources:\n")
        for meta in srcs:
            f.write(
                f" - {meta.get('form')} {meta.get('filed_date')} | {meta.get('section_title')} | {os.path.basename(meta.get('source_path',''))}\n"
            )

    print(f"\nWrote full LLM context to: {out_path}")

    # Basic shape assertions
    assert packet.get("ticker") == ticker.upper()
    assert isinstance(ctx, dict)
    for key in ["business", "performance", "liquidity"]:
        assert key in ctx
        assert isinstance(ctx[key], list)
    assert isinstance(srcs, list)
    # At least one snippet overall is expected
    total_snippets = sum(len(v) for v in ctx.values())
    assert total_snippets >= 1

