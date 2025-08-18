from __future__ import annotations

"""Workflow orchestrator skeleton for development.

- End-to-end steps for a ticker:
  1) Fetch basics (StockClient), TA (TechnicalAnalyzer), and news (NewsClient via tools facade)
  2) Build FinGPT prompt (condensed)
  3) Call FinGPT forecaster API (stubbed here)
  4) Optionally query RAG index for filings (if exists) and attach bullets

In production, split ETL for filings/embeddings/indices into scheduled jobs.
"""

from typing import Dict, Any, List, Optional
import os
from datetime import datetime, timedelta, timezone

from src.tools.tools import get_stock_data, analyze_technical_indicators, get_news
from src.tools.sec_filings import list_local_filings
from src.tools.rag import build_indices_for_ticker_by_form, build_llm_context
from src.models.openai_client import OpenAIClient
from src.models.fingpt_client import FinGPTClient
from src.tools.stock_data import StockClient


def _format_basics_line(basics: Dict[str, Any]) -> Optional[str]:
    if not basics:
        return None
    return (
        f"Basics: {basics.get('company_name','')} | {basics.get('sector','')}, {basics.get('industry','')} | "
        f"{basics.get('current_price')} {basics.get('currency','')} | MktCap {basics.get('market_cap')} | "
        f"52w {basics.get('low_52w')}–{basics.get('high_52w')} | 1d {basics.get('price_change_1d')}%"
    )


def _get_price_window_sentence(ticker: str, days: int = 30) -> str:
    client = StockClient()
    hist = client.get_history(ticker, period=f"{max(days, 7)}d")
    if hist is None or getattr(hist, "empty", True):
        return ""
    df = hist.dropna(subset=["Close"]).copy()
    if df.empty:
        return ""
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    start_price = float(df.loc[df.index.min(), "Close"])  # type: ignore[index]
    end_price = float(df.loc[df.index.max(), "Close"])    # type: ignore[index]
    direction = "increased" if end_price >= start_price else "decreased"
    name = ticker.upper()
    return (
        f"From {start_date} to {end_date}, {name}'s stock price {direction} from "
        f"{start_price:.2f} to {end_price:.2f}. Company news during this period are listed below:"
    )


def _format_basic_financials(basics: Dict[str, Any]) -> List[str]:
    lines: List[str] = ["[Basic Financials]:"]
    mapping = [
        ("P/E Ratio", basics.get("pe_ratio")),
        ("P/B Ratio", basics.get("pb_ratio")),
        ("Dividend Yield", basics.get("dividend_yield")),
        ("Market Cap", basics.get("market_cap")),
        ("Shares Outstanding", basics.get("shares_outstanding")),
    ]
    for label, val in mapping:
        if val is None:
            continue
        try:
            if isinstance(val, (int, float)):
                lines.append(f"{label}: {val}")
            else:
                lines.append(f"{label}: {val}")
        except Exception:
            continue
    return lines


def build_fingpt_prompt(
    ticker: str,
    *,
    basics: Dict[str, Any],
    ta: Dict[str, Any],
    news: Dict[str, Any],
    price_window_days: int = 30,
) -> str:
    name = basics.get("company_name", ticker.upper())
    industry = basics.get("industry", basics.get("sector", ""))
    ipo = basics.get("ipo", "")
    mcap = basics.get("market_cap", "")
    currency = basics.get("currency", "")
    shares = basics.get("shares_outstanding", "")
    country = basics.get("country", "")
    exchange = basics.get("exchange", "")
    symbol = ticker.upper()

    intro = (
        f"[Company Introduction]:\n\n"
        f"{name} is a leading entity in the {industry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. "
        f"As of today, {name} has a market capitalization of {mcap} in {currency}, with {shares} shares outstanding. {name} operates primarily in the {country}, trading under the ticker {symbol} on the {exchange}. "
        f"As a dominant force in the {industry} space, the company continues to innovate and drive progress within the industry.\n\n"
    )

    price_sentence = _get_price_window_sentence(symbol, days=price_window_days)

    # Company News section in requested style with headline/summary blocks
    news_lines: List[str] = ["[Company News]:"]
    comp = news.get("company_news", {}).get("articles", []) if news else []
    for a in comp[:3]:
        news_lines.append(f"[Headline]: {a.get('headline','')}")
        news_lines.append(f"[Summary]: {a.get('summary','')}")
        news_lines.append("")

    # Market News section (bulleted)
    market_lines: List[str] = []
    mkt = news.get("market_news", {}).get("articles", []) if news else []
    if mkt:
        market_lines.append("[Market News]:")
        for a in mkt[:3]:
            market_lines.append(f"- {a.get('source','')}: {a.get('headline','')} {a.get('summary','')}")

    # Basic financials
    fin_lines = _format_basic_financials(basics)

    # Technical Analysis section
    tech_lines: List[str] = []
    if ta:
        tech_lines.append("[Technical Analysis]:")
        rsi = (ta.get('rsi',{}) or {}).get('current')
        macd_sig = (ta.get('macd',{}) or {}).get('signal')
        ma = ta.get('moving_averages',{}) or {}
        verdict = (ta.get('trading_signals',{}) or {}).get('overall_signal')
        conf = (ta.get('trading_signals',{}) or {}).get('confidence')
        if rsi is not None:
            tech_lines.append(f"• RSI: {rsi}")
        if macd_sig is not None:
            tech_lines.append(f"• MACD: {macd_sig}")
        if 'price_vs_sma_50' in ma:
            tech_lines.append(f"• 50-day vs price: {ma.get('price_vs_sma_50')}")
        if verdict is not None:
            tech_lines.append(f"• Verdict: {verdict} ({conf}%)")

    # Dates for final instruction
    curday = datetime.now(timezone.utc).date()
    period = (curday + timedelta(days=7)).isoformat()

    tail = (
        f"\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. "
        f"Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company-related news. "
        f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction.\n"
    )

    parts: List[str] = [intro]
    if price_sentence:
        parts.append(price_sentence + "\n")
    if news_lines:
        parts.extend(news_lines)
    if market_lines:
        parts.append("")
        parts.extend(market_lines)
    if fin_lines:
        parts.append("")
        parts.extend(fin_lines)
        parts.append("")
    if tech_lines:
        parts.extend(tech_lines)
        parts.append("")
    parts.append(tail)
    return "\n".join(p.strip("\n") for p in parts if p is not None)


def _condense_inputs_for_prompt(ticker: str, *, basics: Dict[str, Any], ta: Dict[str, Any], news: Dict[str, Any]) -> str:
    lines: List[str] = [f"Ticker: {ticker.upper()}"]
    b = _format_basics_line(basics)
    if b:
        lines.append(b)
    # Technicals (condensed one-liner)
    if ta:
        ts = ta.get('trading_signals', {}) or {}
        ma = ta.get('moving_averages', {}) or {}
        tline = (
            f"Technicals: RSI {(ta.get('rsi',{}) or {}).get('current')} | "
            f"MACD {(ta.get('macd',{}) or {}).get('signal')} | "
            f"SMA50 vs Px {ma.get('price_vs_sma_50')} | "
            f"Verdict {ts.get('overall_signal')} ({ts.get('confidence')}%)"
        )
        lines.append(tline)
    # News (company then market)
    comp = (news or {}).get('company_news', {}).get('articles', [])
    mkt = (news or {}).get('market_news', {}).get('articles', [])
    if comp:
        lines.append("Company News:")
        for a in comp[:3]:
            lines.append(f"- {a.get('source','')}: {a.get('headline','')} {a.get('summary','')}")
    if mkt:
        lines.append("Market News:")
        for a in mkt[:3]:
            lines.append(f"- {a.get('source','')}: {a.get('headline','')} {a.get('summary','')}")
    return "\n".join(lines)


def _compose_openai_system_prompt() -> str:
    return (
        "You are a seasoned stock market analyst. Given a forecaster's raw output, condensed inputs (basics, news, technicals), "
        "and SEC RAG context, produce a clear, cited recommendation. Output must include: "
        "[Short-Term (1-2 weeks)]: Buy/Hold/Sell with 1-2 positives, 1-2 negatives, key risks, suggested entry/exit and expected magnitude (%). "
        "[Long-Term (6-12 months)]: Buy/Hold/Sell with main drivers and risks. "
        "Be concise, factual, and reflect uncertainty. If citing filings, include the section title and form (e.g., Item 7, 10-K 2024)."
    )


def _format_rag_context_for_prompt(ticker: str, max_per_topic: int = 3) -> str:
    ctx = build_llm_context(ticker, max_per_topic=max_per_topic)
    lines: List[str] = [f"[RAG Context for {ctx.get('ticker')}]\nWarnings: {ctx.get('warnings')}"]
    context = ctx.get("context", {})
    for topic, snippets in context.items():
        lines.append(f"[{topic}] ({len(snippets)})")
        for s in snippets:
            lines.append(f"- {s}")
    # Append sources for transparency
    lines.append("\nSources:")
    for s in ctx.get("sources", [])[:10]:
        lines.append(
            f" - {s.get('form')} {s.get('filed_date')} | {s.get('section_title')} | {os.path.basename(str(s.get('source_path','')))}"
        )
    return "\n".join(lines)


def analyze_stock_with_forecast_and_rag(
    ticker: str,
    *,
    runpod_url: Optional[str] = None,
    openai_model: str = "gpt-5",
) -> Dict[str, Any]:
    """End-to-end: fetch data, call RunPod forecaster, build RAG, and synthesize via OpenAI."""
    basics = get_stock_data(ticker)
    ta = analyze_technical_indicators(ticker)
    news = get_news(ticker, days_back=7)

    condensed = _condense_inputs_for_prompt(ticker, basics=basics, ta=ta, news=news)

    # Prepare FinGPT prompt in requested narrative style (with sections)
    user_prompt = build_fingpt_prompt(ticker, basics=basics, ta=ta, news=news, price_window_days=30)
    client = FinGPTClient(endpoint_url=runpod_url)
    forecaster_raw = client.generate(user_prompt=user_prompt)

    # RAG context over filings
    filings = list_local_filings(ticker)
    if filings:
        build_indices_for_ticker_by_form(ticker, filing_paths=[f.path for f in filings])
    rag_for_prompt = _format_rag_context_for_prompt(ticker, max_per_topic=3)

    # Synthesize with OpenAI
    openai = OpenAIClient(model=openai_model)
    final = openai.chat(
        [
            {"role": "system", "content": _compose_openai_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Ticker: {ticker.upper()}\n\nCondensed Inputs:\n{condensed}\n\nForecaster Output (RunPod):\n{forecaster_raw}\n\n{rag_for_prompt}\n\n"
                    "Please provide the structured recommendation as specified."
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=1200,
    )

    return {
        "ticker": ticker.upper(),
        "condensed_inputs": condensed,
        "forecaster_output": forecaster_raw,
        "rag_context": rag_for_prompt,
        "final_recommendation": final,
    }


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TICKER", "AAPL")
    url = os.getenv("RUNPOD_ENDPOINT_URL")
    result = analyze_stock_with_forecast_and_rag(t, runpod_url=url)
    # Print compact summary
    print({
        "ticker": result.get("ticker"),
        "has_forecaster": bool(result.get("forecaster_output")),
        "rag_len": len(result.get("rag_context", "")),
    })
    print("\n=== Final Recommendation ===\n")
    print(result.get("final_recommendation", ""))
