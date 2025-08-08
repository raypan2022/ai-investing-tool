from src.tools.stock_data import StockDataFetcher

def test_stock_data_fetcher():
    """Test basic functionality of StockDataFetcher"""
    
    # Initialize fetcher
    fetcher = StockDataFetcher(config={})
    
    # Test with a reliable stock (AAPL)
    result = fetcher.get_stock_info("AAPL")
    
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
        
        print(f"✅ Test passed for {result['company_name']}")
        print(f"   Price: ${result['current_price']:.2f}")
        print(f"   Sector: {result['sector']}")
    else:
        print(f"❌ API Error: {result['error']}")

def test_stock_history():
    """Test historical data fetching"""
    fetcher = StockDataFetcher(config={})
    hist = fetcher.get_stock_history("AAPL", "1mo")
    
    assert hist is not None
    if not hist.empty:
        assert 'Close' in hist.columns
        print(f"✅ History test passed - got {len(hist)} days of data")
    else:
        print("❌ History test failed - empty DataFrame")

if __name__ == "__main__":
    print("Testing StockDataFetcher...")
    test_stock_data_fetcher()
    test_stock_history()
    print("Tests complete!")
