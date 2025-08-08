"""Financial analysis tools for LangGraph ReAct agent"""

import yfinance as yf
from typing import Dict, Any, List
from .stock_data import StockDataFetcher
from .technical import TechnicalAnalyzer


def get_stock_data(ticker: str) -> Dict[str, Any]:
    """Get comprehensive stock data including price, fundamentals, and company info.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    """
    fetcher = StockDataFetcher({})
    result = fetcher.get_stock_info(ticker)
    
    # Clean up the result for LLM consumption
    if 'error' not in result:
        return {
            'ticker': result['ticker'],
            'company_name': result['company_name'],
            'current_price': result['current_price'],
            'market_cap': result['market_cap'],
            'pe_ratio': result['pe_ratio'],
            'pb_ratio': result['pb_ratio'],
            'dividend_yield': result['dividend_yield'],
            'sector': result['sector'],
            'industry': result['industry'],
            'price_change_1d': result['price_change_1d'],
            'volume': result['volume'],
            'description': result['description'][:200] + "..." if len(result['description']) > 200 else result['description']
        }
    else:
        return result


def analyze_technical_indicators(ticker: str) -> Dict[str, Any]:
    """Perform comprehensive technical analysis including RSI, MACD, moving averages, and more.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    """
    try:
        # Get historical data using our StockDataFetcher
        fetcher = StockDataFetcher({})
        hist_data = fetcher.get_stock_history(ticker, period="1y")
        
        if hist_data.empty:
            return {"error": f"No historical data available for {ticker}"}
        
        # Use the comprehensive TechnicalAnalyzer
        analyzer = TechnicalAnalyzer({})
        analysis = analyzer.analyze_stock(ticker, hist_data)
        
        # Extract key insights for LLM
        if 'error' not in analysis:
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
        else:
            return analysis
            
    except Exception as e:
        return {"error": f"Technical analysis failed for {ticker}: {str(e)}"}


def get_news_sentiment(ticker: str, days_back: int = 7) -> Dict[str, Any]:
    """Analyze recent news sentiment for a stock.
    
    Args:
        ticker: Stock ticker symbol
        days_back: Number of days to look back for news
    """
    # Placeholder - will be implemented with actual news API
    return {
        "ticker": ticker,
        "sentiment_score": 0.0,
        "sentiment": "neutral",
        "news_count": 0,
        "days_analyzed": days_back,
        "summary": "News sentiment analysis not yet implemented",
        "status": "placeholder"
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
    get_news_sentiment,
    query_financial_documents
]