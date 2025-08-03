from src.config import Config
from src.data.stock_data import StockDataFetcher
import json

def test_stock_data():
    """Test the stock data fetcher"""
    config = Config()
    fetcher = StockDataFetcher(config)
    
    # Test with a few popular stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    for ticker in test_tickers:
        print(f"\n{'='*50}")
        print(f"Testing {ticker}")
        print(f"{'='*50}")
        
        # Get stock info
        info = fetcher.get_stock_info(ticker)
        
        # Print results in a nice format
        if 'error' not in info:
            print(f"Company: {info['company_name']}")
            print(f"Current Price: ${info['current_price']:.2f}")
            print(f"Market Cap: ${info['market_cap']:,.0f}")
            print(f"P/E Ratio: {info['pe_ratio']:.2f}")
            print(f"Sector: {info['sector']}")
            print(f"1-Day Change: {info['price_change_1d']:.2%}")
        else:
            print(f"Error: {info['error']}")
        
        print()

if __name__ == "__main__":
    test_stock_data() 