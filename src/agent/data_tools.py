"""
Data Review Tools for Investment Analysis

This module provides tools for reviewing stock data and technical analysis
that can be used by the ReAct agent.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def review_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Get current stock data and fundamentals
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing stock data and fundamentals
    """
    try:
        # Import here to avoid circular imports
        from src.data.stock_data import StockDataFetcher
        import config
        
        # Initialize stock data fetcher
        stock_fetcher = StockDataFetcher(config)
        
        # Get stock information
        stock_info = stock_fetcher.get_stock_info(ticker)
        
        # Get recent price history
        stock_history = stock_fetcher.get_stock_history(ticker, period="6mo")
        
        # Calculate additional metrics
        if not stock_history.empty:
            # Price changes
            current_price = stock_history['Close'].iloc[-1]
            price_1d = stock_history['Close'].pct_change().iloc[-1] * 100
            price_5d = (current_price / stock_history['Close'].iloc[-5] - 1) * 100 if len(stock_history) >= 5 else 0
            price_20d = (current_price / stock_history['Close'].iloc[-20] - 1) * 100 if len(stock_history) >= 20 else 0
            
            # Volume analysis
            avg_volume = stock_history['Volume'].mean()
            current_volume = stock_history['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility
            daily_returns = stock_history['Close'].pct_change().dropna()
            volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
            
            # Add calculated metrics
            stock_info.update({
                'price_changes': {
                    '1d': price_1d,
                    '5d': price_5d,
                    '20d': price_20d
                },
                'volume_analysis': {
                    'current_volume': current_volume,
                    'avg_volume': avg_volume,
                    'volume_ratio': volume_ratio
                },
                'volatility': volatility,
                'last_updated': datetime.now().isoformat()
            })
        
        logger.info(f"Successfully retrieved stock data for {ticker}")
        return stock_info
        
    except Exception as e:
        logger.error(f"Error reviewing stock data for {ticker}: {e}")
        return {
            'error': f"Failed to retrieve stock data for {ticker}: {str(e)}",
            'ticker': ticker.upper()
        }


def review_technical_analysis(ticker: str) -> Dict[str, Any]:
    """
    Get technical indicators and analysis
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing technical analysis results
    """
    try:
        # Import here to avoid circular imports
        from src.data.stock_data import StockDataFetcher
        from src.analysis.technical_analysis import TechnicalAnalyzer
        import config
        
        # Get stock history
        stock_fetcher = StockDataFetcher(config)
        stock_history = stock_fetcher.get_stock_history(ticker, period="6mo")
        
        if stock_history.empty:
            return {
                'error': f"No historical data available for {ticker}",
                'ticker': ticker.upper()
            }
        
        # Perform technical analysis
        technical_analyzer = TechnicalAnalyzer(config)
        technical_analysis = technical_analyzer.analyze_stock(ticker, stock_history)
        
        # Add summary metrics for easy access
        summary = {
            'overall_signal': technical_analysis.get('overall_signal', 'HOLD'),
            'confidence': technical_analysis.get('confidence', 50),
            'current_price': stock_history['Close'].iloc[-1],
            'key_indicators': {
                'rsi': technical_analysis.get('rsi', {}).get('current', 50),
                'macd_signal': technical_analysis.get('macd', {}).get('signal', 'neutral'),
                'price_vs_sma20': technical_analysis.get('moving_averages', {}).get('price_vs_sma20', 0),
                'bollinger_position': technical_analysis.get('bollinger_bands', {}).get('position', 'middle')
            },
            'risk_metrics': {
                'volatility': technical_analysis.get('advanced_statistics', {}).get('volatility_analysis', {}).get('annualized_volatility', 0),
                'sharpe_ratio': technical_analysis.get('advanced_statistics', {}).get('risk_metrics', {}).get('sharpe_ratio', 0),
                'max_drawdown': technical_analysis.get('advanced_statistics', {}).get('risk_metrics', {}).get('max_drawdown', 0)
            },
            'investment_strategy': technical_analysis.get('investment_strategy', {}),
            'last_updated': datetime.now().isoformat()
        }
        
        # Add summary to technical analysis
        technical_analysis['summary'] = summary
        
        logger.info(f"Successfully performed technical analysis for {ticker}")
        return technical_analysis
        
    except Exception as e:
        logger.error(f"Error reviewing technical analysis for {ticker}: {e}")
        return {
            'error': f"Failed to perform technical analysis for {ticker}: {str(e)}",
            'ticker': ticker.upper()
        }


def get_tool_descriptions() -> Dict[str, str]:
    """Get descriptions of all data tools for ReAct agent"""
    return {
        'review_stock_data': """Get current stock data, fundamentals, and market metrics. 
        Use this tool when users ask about current stock price, market cap, PE ratio, volume, or basic fundamentals.""",
        
        'review_technical_analysis': """Get technical indicators, signals, and analysis. 
        Use this tool when users ask about technical indicators, price patterns, momentum, or technical signals."""
    }


def format_stock_data_summary(stock_data: Dict[str, Any]) -> str:
    """Format stock data for LLM consumption"""
    if 'error' in stock_data:
        return f"Error: {stock_data['error']}"
    
    summary = f"""
Stock Data for {stock_data.get('ticker', 'Unknown')}:
- Current Price: ${stock_data.get('current_price', 0):.2f}
- Market Cap: ${stock_data.get('market_cap', 0):,.0f}
- PE Ratio: {stock_data.get('pe_ratio', 0):.2f}
- Sector: {stock_data.get('sector', 'Unknown')}
- Industry: {stock_data.get('industry', 'Unknown')}

Price Changes:
- 1 Day: {stock_data.get('price_changes', {}).get('1d', 0):.2f}%
- 5 Days: {stock_data.get('price_changes', {}).get('5d', 0):.2f}%
- 20 Days: {stock_data.get('price_changes', {}).get('20d', 0):.2f}%

Volume Analysis:
- Current Volume: {stock_data.get('volume_analysis', {}).get('current_volume', 0):,.0f}
- Average Volume: {stock_data.get('volume_analysis', {}).get('avg_volume', 0):,.0f}
- Volume Ratio: {stock_data.get('volume_analysis', {}).get('volume_ratio', 1):.2f}

Volatility: {stock_data.get('volatility', 0):.2f}% (annualized)
"""
    return summary


def format_technical_analysis_summary(technical_data: Dict[str, Any]) -> str:
    """Format technical analysis for LLM consumption"""
    if 'error' in technical_data:
        return f"Error: {technical_data['error']}"
    
    summary = technical_data.get('summary', {})
    
    formatted = f"""
Technical Analysis for {technical_data.get('ticker', 'Unknown')}:
- Overall Signal: {summary.get('overall_signal', 'HOLD')}
- Confidence: {summary.get('confidence', 50):.1f}%
- Current Price: ${summary.get('current_price', 0):.2f}

Key Indicators:
- RSI: {summary.get('key_indicators', {}).get('rsi', 50):.1f}
- MACD Signal: {summary.get('key_indicators', {}).get('macd_signal', 'neutral')}
- Price vs SMA20: {summary.get('key_indicators', {}).get('price_vs_sma20', 0):.2f}%
- Bollinger Position: {summary.get('key_indicators', {}).get('bollinger_position', 'middle')}

Risk Metrics:
- Volatility: {summary.get('risk_metrics', {}).get('volatility', 0):.2f}%
- Sharpe Ratio: {summary.get('risk_metrics', {}).get('sharpe_ratio', 0):.2f}
- Max Drawdown: {summary.get('risk_metrics', {}).get('max_drawdown', 0):.2f}%

Investment Strategy:
- Recommended Timeframe: {summary.get('investment_strategy', {}).get('recommended_timeframe', 'medium_term')}
- Primary Driver: {summary.get('investment_strategy', {}).get('primary_driver', 'balanced')}
- Entry Strategy: {summary.get('investment_strategy', {}).get('entry_strategy', 'standard')}
"""
    return formatted 