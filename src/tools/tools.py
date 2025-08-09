"""Financial analysis tools for LangGraph ReAct agent"""

from typing import Dict, Any, List
from .stock_data import StockClient
from .technical import TechnicalAnalyzer
from .news import get_company_and_market_news


def get_stock_data(ticker: str) -> Dict[str, Any]:
    """Get comprehensive stock data including price, fundamentals, and company info.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    """
    client = StockClient()
    try:
        result = client.get_info(ticker)

        # Clean up and project fields for LLM consumption (aligns with StockClient schema)
        description = result.get('description', '') or ''
        if len(description) > 200:
            description = description[:200] + "..."
        return {
            'ticker': result.get('ticker'),
            'company_name': result.get('company_name'),
            'current_price': result.get('current_price'),
            'currency': result.get('currency'),
            'exchange': result.get('exchange'),
            'market_cap': result.get('market_cap'),
            'pe_ratio': result.get('pe_ratio'),
            'pb_ratio': result.get('pb_ratio'),
            'dividend_yield': result.get('dividend_yield'),
            'shares_outstanding': result.get('shares_outstanding'),
            'sector': result.get('sector'),
            'industry': result.get('industry'),
            'price_change_1d': result.get('price_change_1d'),
            'volume': result.get('volume'),
            'avg_volume': result.get('avg_volume'),
            'high_52w': result.get('high_52w'),
            'low_52w': result.get('low_52w'),
            'description': description,
            'last_updated': result.get('last_updated'),
        }
    except Exception as exc:
        return {
            'ticker': ticker.upper(),
            'error': f"Failed to get stock data: {str(exc)}",
        }


def analyze_technical_indicators(ticker: str) -> Dict[str, Any]:
    """Perform comprehensive technical analysis including RSI, MACD, moving averages, and more.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    """
    try:
        # Get historical data using StockClient
        client = StockClient()
        hist_data = client.get_history(ticker, period="1y")
        
        if hist_data.empty:
            raise RuntimeError(f"No historical data available for {ticker}")
        
        # Use the comprehensive TechnicalAnalyzer
        analyzer = TechnicalAnalyzer({})
        analysis = analyzer.analyze_stock(ticker, hist_data)

        # Extract key insights for LLM
        summary = {
            'ticker': analysis['ticker'],
            'current_price': analysis['current_price'],
            'rsi': analysis.get('rsi', {}),
            'macd': analysis.get('macd', {}),
            'moving_averages': analysis.get('moving_averages', {}),
            'bollinger_bands': analysis.get('bollinger_bands', {}),
            'trading_signals': {
                'overall_signal': analysis.get('overall_signal', 'HOLD'),
                'confidence': analysis.get('confidence', 50),
                'buy_signals': analysis.get('buy_signals', []),
                'sell_signals': analysis.get('sell_signals', [])
            },
            'technical_summary': analysis.get('technical_summary', {}),
            'support_resistance': analysis.get('pivot_points', {}),
            'volume_analysis': analysis.get('volume', {})
        }
        return summary
            
    except Exception as e:
        return {"error": f"Technical analysis failed for {ticker}: {str(e)}"}


def get_news(ticker: str, days_back: int = 7) -> Dict[str, Any]:
    """Fetch concise company and market news using Finnhub-backed helper.

    Note: sentiment fields removed by request; returns only news blocks.
    """
    try:
        result = get_company_and_market_news(
            ticker,
            days_back=days_back,
            company_limit=3,
            market_limit=3,
        )
        return {
            "ticker": result.get("ticker", ticker.upper()),
            "days_analyzed": days_back,
            "company_news": result.get("company", {}),
            "market_news": result.get("market", {}),
        }
    except Exception as exc:
        return {
            "ticker": ticker.upper(),
            "news_count": 0,
            "days_analyzed": days_back,
            "summary": f"News fetch failed: {str(exc)}",
            "status": "error",
        }


def query_financial_documents(ticker: str, query: str) -> Dict[str, Any]:
    """Query SEC filings and financial documents using RAG.
    
    Args:
        ticker: Stock ticker symbol
        query: Question about the company's financials
    """
    # Placeholder - will be implemented with RAG system
    return {
        "ticker": ticker,
        "query": query,
        "answer": "RAG document querying not yet implemented",
        "sources": [],
        "confidence": 0.0,
        "status": "placeholder"
    }


# List of all available tools for LangGraph
TOOLS = [
    get_stock_data,
    analyze_technical_indicators,
    get_news,
    query_financial_documents
]