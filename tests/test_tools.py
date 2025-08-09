import os
from dotenv import load_dotenv

from src.tools.stock_data import StockClient
from src.tools.tools import (
    get_stock_data,
    analyze_technical_indicators,
    get_news,
)

load_dotenv()

def test_stock_client_info():
    """Test basic functionality of StockClient.get_info"""
    client = StockClient()
    result = client.get_info("AAPL")
    
    # Basic assertions
    assert result is not None
    assert 'ticker' in result
    assert result['ticker'] == 'AAPL'
    
    if 'error' not in result:
        # Check required fields are present
        required_fields = ['company_name', 'current_price', 'market_cap', 'sector']
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
        
        # Check data types
        assert isinstance(result['current_price'], (int, float))
        assert result['current_price'] > 0
        
        print(f"✅ Info test passed for {result['company_name']}")
        print(f"   Price: ${result['current_price']:.2f}")
        print(f"   Sector: {result['sector']}")
    else:
        print(f"❌ API Error: {result['error']}")

def test_stock_client_history():
    """Test historical data fetching via StockClient"""
    client = StockClient()
    hist = client.get_history("AAPL", period="1mo")
    
    assert hist is not None
    if not hist.empty:
        assert 'Close' in hist.columns
        print(f"✅ History test passed - got {len(hist)} rows of data")
    else:
        print("❌ History test failed - empty DataFrame")

if __name__ == "__main__":
    print("Testing StockClient...")
    test_stock_client_info()
    test_stock_client_history()
    print("Tests complete!")


def test_tools_get_stock_data_print():
    """Integration: tools.get_stock_data prints a concise snapshot."""
    res = get_stock_data("AAPL")
    assert isinstance(res, dict)
    print("\n[tools.get_stock_data] AAPL:")
    for k in [
        "company_name",
        "current_price",
        "currency",
        "market_cap",
        "pe_ratio",
        "pb_ratio",
        "dividend_yield",
        "price_change_1d",
        "volume",
        "high_52w",
        "low_52w",
    ]:
        print(f"  {k}: {res.get(k)}")


def test_tools_analyze_technical_print():
    """Integration: tools.analyze_technical_indicators prints summary."""
    res = analyze_technical_indicators("AAPL")
    assert isinstance(res, dict)
    print("\n[tools.analyze_technical_indicators] AAPL:")
    ts = res.get("trading_signals", {})
    ma = res.get("moving_averages", {})
    rsi = res.get("rsi", {})
    macd = res.get("macd", {})
    print(f"  overall_signal: {ts.get('overall_signal')} (confidence={ts.get('confidence')})")
    print(f"  RSI: {rsi}")
    print(f"  MACD: {macd}")
    print(f"  MAs: price_vs_sma_50={ma.get('price_vs_sma_50')} sma_50={ma.get('sma_50')}")


def test_tools_get_news_print():
    """Integration: tools.get_news prints company and market headlines.

    Skips if FINNHUB_API_KEY is not set.
    """
    if not os.getenv("FINNHUB_API_KEY"):
        import pytest
        pytest.skip("FINNHUB_API_KEY not set")

    res = get_news("AAPL", days_back=7)
    assert isinstance(res, dict)
    print("\n[tools.get_news] AAPL:")
    comp = res.get("company_news", {}).get("articles", [])
    mkt = res.get("market_news", {}).get("articles", [])
    print("  Company news:")
    for a in comp:
        print(f"   - {a.get('headline','')} | {a.get('source','')}")
        if a.get('summary'): print(f"     {a.get('summary')}")
    print("  Market news:")
    for a in mkt:
        print(f"   - {a.get('headline','')} | {a.get('source','')}")
        if a.get('summary'): print(f"     {a.get('summary')}")
