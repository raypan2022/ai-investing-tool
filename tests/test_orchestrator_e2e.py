import os
import pytest
from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env at import time
load_dotenv()

from src.workflow.orchestrator import (
    analyze_stock_with_forecast_and_rag,
    build_fingpt_prompt,
    _compose_openai_system_prompt,
)
from src.tools.tools import get_stock_data, analyze_technical_indicators, get_news


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

    ticker = os.getenv("E2E_TICKER", "SNOW")

    # Build and print the exact FinGPT prompt
    basics = get_stock_data(ticker)
    ta = analyze_technical_indicators(ticker)
    news = get_news(ticker, days_back=7)
    fingpt_prompt = build_fingpt_prompt(ticker, basics=basics, ta=ta, news=news, price_window_days=30)
    print("\n[FinGPT Prompt]:\n" + fingpt_prompt)

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
    # Reconstruct and print the OpenAI messages used
    user_msg = (
        f"Ticker: {ticker.upper()}\n\nCondensed Inputs:\n{result.get('condensed_inputs','')}\n\n"
        f"Forecaster Output (RunPod):\n{result.get('forecaster_output','')}\n\n"
        f"{result.get('rag_context','')}\n\n"
        "Please provide the structured recommendation as specified."
    )
    print("\n[OpenAI System Prompt]:\n" + _compose_openai_system_prompt())
    print("\n[OpenAI User Prompt]:\n" + user_msg)
    print("\n[Final Recommendation]:\n" + final)


