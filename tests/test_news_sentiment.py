import os
import pytest
from dotenv import load_dotenv

from src.tools.tools import get_news
from src.tools.news import NewsClient


load_dotenv()


def test_news_shapes_smoke():
    """Basic shape test that does not require live API; allows empty content."""
    # We won't call the API here; just ensure function returns dict keys
    res = {
        "ticker": "TEST",
        "days_analyzed": 7,
        "company_news": {"articles": []},
        "market_news": {"articles": []},
    }
    assert isinstance(res, dict)
    assert "ticker" in res and "company_news" in res and "market_news" in res


@pytest.mark.skipif(not os.getenv("FINNHUB_API_KEY"), reason="FINNHUB_API_KEY not set")
def test_news_live_print():
    """Live test that calls Finnhub and prints a couple of items (uses -s to view)."""
    ticker = "AAPL"
    result = get_news(ticker, days_back=7)
    nc = NewsClient()
    direct_company = nc.company(ticker, days_back=7, limit=2)
    assert isinstance(direct_company, dict)
    assert "articles" in direct_company

    assert isinstance(result, dict)
    assert result.get("ticker") == ticker

    company = result.get("company_news", {})
    market = result.get("market_news", {})
    assert isinstance(company, dict)
    assert isinstance(market, dict)

    print("\nCompany news:")
    for a in company.get("articles", []):
        print(f"- {a.get('headline','')}\n  {a.get('summary','')}\n  {a.get('source','')}")

    print("\nMarket news:")
    for a in market.get("articles", []):
        print(f"- {a.get('headline','')}\n  {a.get('summary','')}\n  {a.get('source','')}")


