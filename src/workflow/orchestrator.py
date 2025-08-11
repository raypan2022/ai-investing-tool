from __future__ import annotations

"""Workflow orchestrator skeleton for development.

- End-to-end steps for a ticker:
  1) Fetch basics (StockClient), TA (TechnicalAnalyzer), and news (NewsClient via tools facade)
  2) Build FinGPT prompt (condensed)
  3) Call FinGPT forecaster API (stubbed here)
  4) Optionally query RAG index for filings (if exists) and attach bullets

In production, split ETL for filings/embeddings/indices into scheduled jobs.
"""

from typing import Dict, Any

from src.tools.tools import get_stock_data, analyze_technical_indicators, get_news
from src.tools.sec_filings import fetch_and_cache_filings, list_local_filings
from src.tools.rag import build_indices_for_ticker_by_form, query_index


def build_forecaster_prompt(ticker: str, *, basics: Dict[str, Any], ta: Dict[str, Any], news: Dict[str, Any]) -> str:
    # Minimal prompt composer; in practice format with tight, single-line bullets
    lines = [f"Ticker: {ticker}"]
    if basics and 'company_name' in basics:
        lines.append(
            f"Basics: {basics.get('company_name','')} | {basics.get('sector','')}, {basics.get('industry','')} | "
            f"{basics.get('current_price')} {basics.get('currency','')} | MktCap {basics.get('market_cap')} | "
            f"52w {basics.get('low_52w')}–{basics.get('high_52w')} | 1d {basics.get('price_change_1d')}%"
        )
    comp = news.get('company_news', {}).get('articles', []) if news else []
    mkt = news.get('market_news', {}).get('articles', []) if news else []
    if comp:
        lines.append("Company News:")
        for a in comp[:3]:
            lines.append(f"- {a.get('source','')}: {a.get('headline','')} {a.get('summary','')}")
    if mkt:
        lines.append("Market News:")
        for a in mkt[:3]:
            lines.append(f"- {a.get('source','')}: {a.get('headline','')} {a.get('summary','')}")
    if ta:
        ts = ta.get('trading_signals', {})
        ma = ta.get('moving_averages', {})
        lines.append(
            f"Technicals: Trend Px vs SMA50={ma.get('price_vs_sma_50')} | RSI {ta.get('rsi',{}).get('current')} | "
            f"MACD {ta.get('macd',{}).get('signal')} | Verdict {ts.get('overall_signal')} ({ts.get('confidence')}%)"
        )
    lines.append("Instruction: Predict next-week direction (up/down/flat) with brief rationale.")
    return "\n".join(lines)


def run_end_to_end_analysis(ticker: str) -> Dict[str, Any]:
    basics = get_stock_data(ticker)
    ta = analyze_technical_indicators(ticker)
    news = get_news(ticker, days_back=7)

    prompt = build_forecaster_prompt(ticker, basics=basics, ta=ta, news=news)

    # Stub forecaster call (replace with real HTTP call)
    fingpt_prediction = {
        "direction": "up",
        "confidence": 0.62,
        "rationale": "Momentum improving, macro supportive; watch key resistance.",
    }

    # RAG: if index exists, query for risk/opportunity bullets
    filings = list_local_filings(ticker)
    index_path = None
    if filings:
        # Build form-specific indices
        idx_paths = build_indices_for_ticker_by_form(ticker, filing_paths=[f.path for f in filings])
        index_path = idx_paths.get("10-Q") or idx_paths.get("10-K")
        # Query both and tag source form in meta
        hits_q = query_index(ticker, query=f"{ticker} guidance liquidity leverage outlook", form="10-Q")
        hits_k = query_index(ticker, query=f"{ticker} guidance liquidity leverage outlook", form="10-K")
        # Add label for form provenance to each hit
        for h in hits_q:
            h.setdefault("meta", {})["source_form"] = "10-Q"
        for h in hits_k:
            h.setdefault("meta", {})["source_form"] = "10-K"
        rag_hits = (hits_q + hits_k)
    else:
        rag_hits = []

    return {
        "ticker": ticker.upper(),
        "prompt": prompt,
        "forecast": fingpt_prediction,
        "rag_hits": rag_hits,
        "index_built": bool(index_path),
    }


