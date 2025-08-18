import os
import pytest


def test_analyze_stock_with_forecast_and_rag_end_to_end(monkeypatch):
    import src.workflow.orchestrator as orch

    # Ensure FinGPT client sees an endpoint, but we stub the network call anyway
    os.environ["RUNPOD_ENDPOINT_URL"] = "https://example.com"

    # Stub inputs (no network)
    def _fake_get_stock_data(t):
        return {
            "company_name": "NVIDIA",
            "sector": "Technology",
            "industry": "Semiconductors",
            "current_price": 500.0,
            "currency": "USD",
            "market_cap": 1_500_000_000_000,
            "low_52w": 300.0,
            "high_52w": 550.0,
            "price_change_1d": 1.2,
            "shares_outstanding": 2_470_000_000,
            "exchange": "NASDAQ",
            "country": "United States",
        }

    def _fake_analyze_ta(t):
        return {
            "rsi": {"current": 62},
            "macd": {"signal": "bullish"},
            "moving_averages": {"price_vs_sma_50": "+3.2%"},
            "trading_signals": {"overall_signal": "BUY", "confidence": 68},
        }

    def _fake_get_news(t, days_back=7):
        return {
            "company_news": {
                "articles": [
                    {"headline": "Strong Q2 Earnings", "summary": "Revenue +70% YoY."},
                    {"headline": "New AI GPU", "summary": "H200 boosts LLM perf."},
                ]
            },
            "market_news": {
                "articles": [
                    {"source": "Bloomberg", "headline": "AI demand remains robust", "summary": "Capex rising."}
                ]
            },
        }

    def _fake_build_llm_context(ticker, max_per_topic=3):
        return {
            "ticker": ticker.upper(),
            "context": {"business": ["Sample biz snippet"], "performance": [], "liquidity": []},
            "sources": [],
            "warnings": [],
        }

    # Monkeypatch orchestrator module names directly
    monkeypatch.setattr(orch, "get_stock_data", _fake_get_stock_data, raising=True)
    monkeypatch.setattr(orch, "analyze_technical_indicators", _fake_analyze_ta, raising=True)
    monkeypatch.setattr(orch, "get_news", _fake_get_news, raising=True)
    monkeypatch.setattr(orch, "build_llm_context", _fake_build_llm_context, raising=True)

    # Stub FinGPT and OpenAI clients
    monkeypatch.setattr(orch.FinGPTClient, "generate", lambda self, user_prompt: "[Forecaster] BUY short-term" , raising=True)
    monkeypatch.setattr(orch.OpenAIClient, "chat", lambda self, messages, temperature=0.2, max_tokens=1200: "[Final] BUY short-term, HOLD long-term" , raising=True)

    out = orch.analyze_stock_with_forecast_and_rag("NVDA")

    # Assertions
    assert out["ticker"] == "NVDA"
    assert "condensed_inputs" in out and isinstance(out["condensed_inputs"], str)
    assert "forecaster_output" in out and out["forecaster_output"].startswith("[Forecaster]")
    assert "rag_context" in out and isinstance(out["rag_context"], str)
    assert "final_recommendation" in out and out["final_recommendation"].startswith("[Final]")

    # Useful print for visibility during -s runs
    print("\n[Final Recommendation]:\n" + out["final_recommendation"]) 


