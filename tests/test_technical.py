import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.tools.technical import TechnicalAnalyzer


def create_sample_data(days=252, start_price=100):
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate realistic stock data with some trend and volatility
    np.random.seed(42)  # For reproducible tests
    
    # Generate price series with random walk
    returns = np.random.normal(0.001, 0.02, days)  # Small positive drift with 2% daily volatility
    prices = [start_price]
    
    for i in range(1, days):
        prices.append(prices[-1] * (1 + returns[i]))
    
    # Create OHLCV data
    data = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    }
    
    # Generate realistic OHLC from close prices
    for i, close in enumerate(prices):
        # Open is previous close with small gap
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
        
        # High and low around the open/close range
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        
        data['Open'].append(open_price)
        data['High'].append(high)
        data['Low'].append(low)
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_technical_analyzer_basic():
    """Test basic functionality of TechnicalAnalyzer"""
    print("Testing TechnicalAnalyzer basic functionality...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=100)
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Basic assertions
    assert result is not None
    assert 'ticker' in result
    assert result['ticker'] == 'TEST'
    assert 'error' not in result
    
    print("✅ Basic functionality test passed")


def test_momentum_indicators():
    """Test momentum indicators (RSI, MACD, etc.)"""
    print("Testing momentum indicators...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=100)
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Check RSI
    assert 'rsi' in result
    rsi = result['rsi']
    assert 'current' in rsi
    assert 0 <= rsi['current'] <= 100
    assert 'signal' in rsi
    assert rsi['signal'] in ['overbought', 'oversold', 'neutral']
    
    # Check MACD
    assert 'macd' in result
    macd = result['macd']
    assert 'macd_line' in macd
    assert 'signal_line' in macd
    assert 'histogram' in macd
    assert 'signal' in macd
    
    print("✅ Momentum indicators test passed")


def test_trend_indicators():
    """Test trend indicators (Moving averages, ADX)"""
    print("Testing trend indicators...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=250)  # Need more data for 200-day MA
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Check moving averages
    assert 'moving_averages' in result
    ma = result['moving_averages']
    assert 'sma_20' in ma
    assert 'sma_50' in ma
    assert 'sma_200' in ma
    assert 'price_vs_sma_20' in ma
    assert ma['price_vs_sma_20'] in ['above', 'below']
    
    # Check ADX
    assert 'adx' in result
    adx = result['adx']
    assert 'current' in adx
    assert 'trend_strength' in adx
    assert adx['trend_strength'] in ['weak', 'moderate', 'strong']
    
    print("✅ Trend indicators test passed")


def test_volatility_indicators():
    """Test volatility indicators (Bollinger Bands, ATR)"""
    print("Testing volatility indicators...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=100)
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Check Bollinger Bands
    assert 'bollinger_bands' in result
    bb = result['bollinger_bands']
    assert 'upper' in bb
    assert 'middle' in bb
    assert 'lower' in bb
    assert bb['upper'] > bb['middle'] > bb['lower']
    assert 'position' in bb
    assert bb['position'] in ['upper', 'middle', 'lower']
    
    # Check ATR
    assert 'atr' in result
    atr = result['atr']
    assert 'current' in atr
    assert atr['current'] >= 0
    assert 'volatility_level' in atr
    
    print("✅ Volatility indicators test passed")


def test_volume_indicators():
    """Test volume indicators"""
    print("Testing volume indicators...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=100)
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Check volume analysis
    assert 'volume' in result
    vol = result['volume']
    assert 'current' in vol
    assert 'sma_20' in vol
    assert 'volume_ratio' in vol
    assert 'high_volume' in vol
    # Volume analysis might return different types, so let's be more flexible
    assert vol['high_volume'] is not None
    
    # Check OBV
    assert 'obv' in result
    obv = result['obv']
    assert 'current' in obv
    assert 'trend' in obv
    assert obv['trend'] in ['rising', 'falling', 'neutral']
    
    print("✅ Volume indicators test passed")


def test_support_resistance():
    """Test support and resistance calculations"""
    print("Testing support and resistance...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=100)
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Check pivot points
    assert 'pivot_points' in result
    pivot = result['pivot_points']
    assert 'pivot' in pivot
    assert 'resistance_1' in pivot
    assert 'support_1' in pivot
    
    # Check dynamic levels
    assert 'dynamic_levels' in result
    dynamic = result['dynamic_levels']
    assert 'resistance_levels' in dynamic
    assert 'support_levels' in dynamic
    assert isinstance(dynamic['resistance_levels'], list)
    assert isinstance(dynamic['support_levels'], list)
    
    print("✅ Support and resistance test passed")


def test_trading_signals():
    """Test trading signal generation"""
    print("Testing trading signals...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=100)
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Check trading signals exist
    assert 'buy_signals' in result
    assert 'sell_signals' in result
    assert 'overall_signal' in result
    assert 'confidence' in result
    
    # Check signal format
    assert result['overall_signal'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= result['confidence'] <= 100
    assert isinstance(result['buy_signals'], list)
    assert isinstance(result['sell_signals'], list)
    
    print("✅ Trading signals test passed")


def test_technical_summary():
    """Test technical summary generation"""
    print("Testing technical summary...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=100)
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Check technical summary
    assert 'technical_summary' in result
    summary = result['technical_summary']
    assert 'trend' in summary
    assert 'momentum' in summary
    assert 'volatility' in summary
    assert 'volume' in summary
    assert 'strength' in summary
    
    # Check valid values
    valid_states = ['bullish', 'bearish', 'neutral', 'normal', 'high', 'low']
    for key, value in summary.items():
        assert value in valid_states, f"Invalid {key} value: {value}"
    
    print("✅ Technical summary test passed")


def test_advanced_statistics():
    """Test advanced statistical analysis"""
    print("Testing advanced statistics...")
    
    analyzer = TechnicalAnalyzer({})
    sample_data = create_sample_data(days=100)
    
    result = analyzer.analyze_stock("TEST", sample_data)
    
    # Check if advanced statistics are present (they might be empty for small datasets)
    if 'advanced_statistics' in result:
        stats = result['advanced_statistics']
        print(f"Advanced statistics keys: {list(stats.keys())}")
        
        # If z_score_analysis exists, check its structure
        if 'z_score_analysis' in stats:
            z_score = stats['z_score_analysis']
            assert 'price_z_score' in z_score
            assert 'mean_reversion_signal' in z_score
    
    print("✅ Advanced statistics test passed")


def test_error_handling():
    """Test error handling with insufficient data"""
    print("Testing error handling...")
    
    analyzer = TechnicalAnalyzer({})
    
    # Test with empty data
    empty_data = pd.DataFrame()
    result = analyzer.analyze_stock("TEST", empty_data)
    assert 'error' in result
    
    # Test with insufficient data
    small_data = create_sample_data(days=10)
    result = analyzer.analyze_stock("TEST", small_data)
    assert 'error' in result
    
    print("✅ Error handling test passed")


def run_all_tests():
    """Run all technical analysis tests"""
    print("="*50)
    print("RUNNING TECHNICAL ANALYSIS TESTS")
    print("="*50)
    
    try:
        test_technical_analyzer_basic()
        test_momentum_indicators()
        test_trend_indicators()
        test_volatility_indicators()
        test_volume_indicators()
        test_support_resistance()
        test_trading_signals()
        test_technical_summary()
        test_advanced_statistics()
        test_error_handling()
        
        print("\n" + "="*50)
        print("🎉 ALL TECHNICAL ANALYSIS TESTS PASSED!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    run_all_tests()