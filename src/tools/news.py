from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import finnhub
from .stock_data import StockClient


# Default market headline keywords to surface macro drivers, risks, and insights
DEFAULT_MARKET_KEYWORDS: List[str] = [
    # Central banks & policy
    "Federal Reserve", "Fed", "ECB", "Bank of England", "BoE", "Bank of Japan", "BoJ", "central bank",
    "interest rate", "rate cut", "rate hike", "policy rate",
    # Inflation & growth
    "inflation", "CPI", "PCE", "core PCE", "GDP", "growth", "recession", "soft landing",
    # Labor market
    "jobs", "unemployment", "payrolls", "nonfarm", "NFP",
    # Rates, credit & liquidity
    "Treasury", "yield", "bond market", "credit spread", "liquidity", "volatility", "VIX", "yield curve",
    # Commodities & energy
    "oil", "OPEC", "WTI", "Brent", "energy prices",
    # Geopolitics & trade
    "China", "tariff", "sanction", "trade war", "Middle East", "Ukraine", "conflict",
    # Broad market indices & FX
    "S&P 500", "Nasdaq", "Dow", "U.S. dollar", "dollar", "DXY", "yen", "yuan",
    # Fiscal policy
    "fiscal", "deficit", "government shutdown", "debt ceiling",
    # Surveys
    "PMI", "ISM",
    # Note: sector/thematic terms like 'AI', 'semiconductor', 'banks' are excluded by default
    # to reduce company-specific noise. They can be provided via the `keywords` parameter.
]


def _get_client(api_key: Optional[str] = None) -> finnhub.Client:
    key = api_key or os.getenv("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY is not set.")
    return finnhub.Client(api_key=key)


class NewsClient:
    def __init__(self, api_key: Optional[str] = None):
        self.client = _get_client(api_key)

    def company(self, ticker: str, *, days_back: int = 7, limit: int = 3) -> Dict[str, Any]:
        return fetch_company_news(ticker, days_back=days_back, limit=limit, client=self.client)

    def market(
        self,
        *,
        days_back: int = 7,
        limit: int = 3,
        keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return fetch_market_news(days_back=days_back, limit=limit, client=self.client, keywords=keywords)


def _to_iso(epoch: Any) -> str:
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    except Exception:
        return ""


def fetch_company_news(
    ticker: str,
    *,
    days_back: int = 7,
    limit: int = 3,
    client: Optional[finnhub.Client] = None,
) -> Dict[str, Any]:
    """Fetch recent company news via Finnhub and return minimal fields for FinGPT.

    Raises exceptions on failures; callers should catch.
    """
    fh = client or _get_client()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)
    data = fh.company_news(ticker.upper(), _from=start_dt.strftime("%Y-%m-%d"), to=end_dt.strftime("%Y-%m-%d")) or []

    # Resolve full company name for stricter headline filtering
    company_name = None
    try:
        info = StockClient({}).get_info(ticker)
        if isinstance(info, dict):
            company_name = (info.get("company_name") or "").strip()
    except Exception:
        company_name = None

    # Keep items where headline contains ticker or company name
    ticker_up = ticker.upper()
    articles: List[Dict[str, Any]] = []
    for item in data:
        headline = item.get("headline", "") or ""
        headline_lc = headline.lower()
        company_match = False
        if company_name:
            company_match = company_name.lower() in headline_lc
        if (ticker_up in headline) or company_match:
            articles.append(
                {
                    "headline": headline,
                    "summary": item.get("summary", "") or "",
                    "source": item.get("source", ""),
                }
            )
        if len(articles) >= max(0, limit):
            break

    return {
        "ticker": ticker.upper(),
        "articles": articles,
    }


def fetch_market_news(
    *,
    days_back: int = 7,
    limit: int = 3,
    client: Optional[finnhub.Client] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fetch recent market news via Finnhub and return minimal fields for FinGPT.

    Raises exceptions on failures; callers should catch.
    """
    fh = client or _get_client()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    data = fh.general_news("general", min_id=0) or []

    articles: List[Dict[str, Any]] = []
    # Decide keyword filter: if None -> use defaults; if [] -> no filtering
    eff_keywords = DEFAULT_MARKET_KEYWORDS if keywords is None else keywords

    for item in data:
        dt = item.get("datetime")
        if isinstance(dt, (int, float)) and dt < cutoff.timestamp():
            continue
        headline = item.get("headline", "") or ""
        summary = item.get("summary", "") or ""
        # Optional: filter by keywords in headline for more relevance
        hl_lc = headline.lower()
        if eff_keywords and not any(kw.lower() in hl_lc for kw in eff_keywords):
            continue
        # Heuristic: drop company-specific items — if only a few symbols are tagged, skip
        related_field = (item.get("related") or "").strip()
        if related_field:
            related_syms = [s for s in related_field.split(",") if s]
            if len(related_syms) <= 3:
                continue
        articles.append(
            {
                "headline": headline,
                "summary": summary,
                "source": item.get("source", ""),
            }
        )
        if len(articles) >= limit:
            break

    return {
        "category": "general",
        "articles": articles,
    }


def get_company_and_market_news(
    ticker: str,
    *,
    days_back: int = 7,
    company_limit: int = 3,
    market_limit: int = 3,
) -> Dict[str, Any]:
    """Convenience function combining company and market news in one structure."""
    fh = _get_client()
    company = fetch_company_news(ticker, days_back=days_back, limit=company_limit, client=fh)
    market = fetch_market_news(days_back=days_back, limit=market_limit, client=fh)

    return {
        "ticker": ticker.upper(),
        "company": company,
        "market": market,
    }


