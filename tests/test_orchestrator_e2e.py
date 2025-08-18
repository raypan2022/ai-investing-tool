import os
import pytest
from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env at import time
load_dotenv()

from src.workflow.orchestrator import analyze_stock_with_forecast_and_rag


def test_orchestrator_e2e_prints_recommendation(capsys):
    """End-to-end test (no mocks): requires RUNPOD_ENDPOINT_URL and OPENAI_API_KEY in .env.

    - Loads environment from .env
    - Invokes full orchestrator flow (stock basics, TA, news, FinGPT on RunPod, RAG, OpenAI)
    - Prints final recommendation for visibility during -s runs
    """
    runpod_url = os.getenv("RUNPOD_ENDPOINT_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not runpod_url or not openai_key:
        pytest.skip("RUNPOD_ENDPOINT_URL or OPENAI_API_KEY not set; skipping E2E test")

    ticker = os.getenv("E2E_TICKER", "AAPL")

    result = analyze_stock_with_forecast_and_rag(ticker, runpod_url=runpod_url)

    # Basic structural checks
    assert isinstance(result, dict)
    assert result.get("ticker") == ticker.upper()
    assert isinstance(result.get("condensed_inputs", ""), str)
    assert isinstance(result.get("forecaster_output", ""), str)
    assert isinstance(result.get("rag_context", ""), str)
    final = result.get("final_recommendation", "")
    assert isinstance(final, str) and len(final.strip()) > 0

    # Print for -s visibility
    print(f"\n[E2E Ticker]: {ticker.upper()}")
    print("[Forecaster Output]:\n" + result.get("forecaster_output", ""))
    print("\n[Final Recommendation]:\n" + final)


