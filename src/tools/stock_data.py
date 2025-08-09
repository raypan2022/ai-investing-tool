import logging
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def fetch_stock_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch historical OHLCV data with sane defaults.

    Returns a DataFrame with at least columns: Open, High, Low, Close, Volume.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval, auto_adjust=auto_adjust)
    if not isinstance(hist, pd.DataFrame):
        raise RuntimeError(f"History fetch did not return a DataFrame for {ticker}")
    return hist


def fetch_stock_info(ticker: str) -> Dict[str, Any]:
    """Fetch comprehensive stock information using yfinance, with fallbacks and normalization.

    Raises exceptions on failures; callers should catch.
    """
    stock = yf.Ticker(ticker)

    # Prefer fast_info for realtime-ish fields
    fast_info = getattr(stock, "fast_info", {}) or {}
    # get_info for descriptive metadata
    try:
        info = stock.get_info() or {}
    except Exception:
        info = getattr(stock, "info", {}) or {}

    # Recent history for price change and volume fallback
    hist = fetch_stock_history(ticker, period="2d")

    # Fields with fallbacks
    current_price = _to_float(
        fast_info.get("last_price", info.get("currentPrice", info.get("regularMarketPrice")))
    )
    market_cap = _to_float(fast_info.get("market_cap", info.get("marketCap")))
    pe_ratio = _to_float(info.get("trailingPE"))
    pb_ratio = _to_float(info.get("priceToBook"))
    dividend_yield = _to_float(info.get("dividendYield"))
    shares_outstanding = _to_int(fast_info.get("shares_outstanding", info.get("sharesOutstanding")))
    currency = (fast_info.get("currency") or info.get("currency") or "").upper()
    exchange = info.get("exchange") or info.get("fullExchangeName") or ""

    # Volumes
    last_volume = fast_info.get("last_volume")
    if last_volume is None and isinstance(hist, pd.DataFrame) and not hist.empty and "Volume" in hist.columns:
        last_volume = hist["Volume"].iloc[-1]
    volume = _to_int(last_volume, default=_to_int(info.get("volume")))
    avg_volume = _to_int(info.get("averageVolume"))

    # 52-week range
    high_52w = _to_float(fast_info.get("year_high", info.get("fiftyTwoWeekHigh")))
    low_52w = _to_float(fast_info.get("year_low", info.get("fiftyTwoWeekLow")))

    # Company identity
    company_name = info.get("longName") or info.get("shortName") or "Unknown"
    sector = info.get("sector") or "Unknown"
    industry = info.get("industry") or "Unknown"
    description_full = info.get("longBusinessSummary") or ""
    description = (description_full[:500] + "...") if description_full else ""

    # Price change 1d (percent)
    if isinstance(hist, pd.DataFrame) and len(hist) > 1 and "Close" in hist.columns:
        pct = hist["Close"].pct_change().iloc[-1]
        try:
            price_change_1d = float(pct) if pd.notna(pct) else 0.0
        except Exception:
            price_change_1d = 0.0
    else:
        price_change_1d = 0.0

    result: Dict[str, Any] = {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "current_price": current_price,
        "currency": currency,
        "exchange": exchange,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
        "dividend_yield": dividend_yield,
        "shares_outstanding": shares_outstanding,
        "sector": sector,
        "industry": industry,
        "description": description,
        "price_change_1d": price_change_1d,
        "volume": volume,
        "avg_volume": avg_volume,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    return result


# Backward-compatible OOP wrapper (used elsewhere in the codebase)
class StockDataFetcher:
    """Deprecated: use StockClient instead."""
    def __init__(self, config):
        self.config = config

    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        return fetch_stock_info(ticker)

    def get_stock_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        return fetch_stock_history(ticker, period=period)


class StockClient:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def get_info(self, ticker: str) -> Dict[str, Any]:
        return fetch_stock_info(ticker)

    def get_history(
        self,
        ticker: str,
        *,
        period: str = "1y",
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        return fetch_stock_history(ticker, period=period, interval=interval, auto_adjust=auto_adjust)