import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self, config):
        self.config = config
        
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch comprehensive stock information using yfinance"""
        try:
            print(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            
            # Get basic info
            info = stock.info
            
            # Get current price and basic metrics
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            # Calculate additional metrics
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            dividend_yield = info.get('dividendYield', 0)
            
            # Get recent price history
            hist = stock.history(period="1mo")
            
            result = {
                'ticker': ticker.upper(),
                'company_name': info.get('longName', info.get('shortName', 'Unknown')),
                'current_price': current_price,
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'dividend_yield': dividend_yield,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'description': info.get('longBusinessSummary', '')[:500] + "..." if info.get('longBusinessSummary') else '',
                'price_change_1d': hist['Close'].pct_change().iloc[-1] if len(hist) > 1 else 0,
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'last_updated': datetime.now().isoformat()
            }
            
            print(f"Successfully fetched data for {ticker}")
            return result
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return {
                'error': f"Failed to fetch data for {ticker}: {str(e)}",
                'ticker': ticker.upper()
            }
    
    def get_stock_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data"""
        try:
            stock = yf.Ticker(ticker)
            return stock.history(period=period)
        except Exception as e:
            print(f"Error fetching history for {ticker}: {str(e)}")
            return pd.DataFrame() 