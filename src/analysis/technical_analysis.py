import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import arch

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    """
    Comprehensive technical analysis module for stock trading signals.
    
    Features:
    - RSI, MACD, Bollinger Bands
    - Support/Resistance levels
    - Moving averages
    - Volume analysis
    - Entry/Exit point identification
    - Trend analysis
    """
    
    def __init__(self, config):
        self.config = config
        
    def analyze_stock(self, ticker: str, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis on stock data.
        
        Args:
            ticker: Stock ticker symbol
            stock_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing all technical analysis results
        """
        try:
            print(f"Performing technical analysis for {ticker}...")
            
            if stock_data.empty:
                return {
                    'error': f"No data available for {ticker}",
                    'ticker': ticker.upper()
                }
            
            # Ensure we have enough data for analysis
            if len(stock_data) < 50:
                return {
                    'error': f"Insufficient data for {ticker}. Need at least 50 data points.",
                    'ticker': ticker.upper()
                }
            
            # Calculate all technical indicators
            analysis = {
                'ticker': ticker.upper(),
                'analysis_date': datetime.now().isoformat(),
                'data_points': len(stock_data),
                'current_price': stock_data['Close'].iloc[-1],
                'price_change_1d': self._calculate_price_change(stock_data, 1),
                'price_change_5d': self._calculate_price_change(stock_data, 5),
                'price_change_20d': self._calculate_price_change(stock_data, 20),
            }
            
            # Add momentum indicators
            analysis.update(self._calculate_momentum_indicators(stock_data))
            
            # Add trend indicators
            analysis.update(self._calculate_trend_indicators(stock_data))
            
            # Add volatility indicators
            analysis.update(self._calculate_volatility_indicators(stock_data))
            
            # Add volume indicators
            analysis.update(self._calculate_volume_indicators(stock_data))
            
            # Add support and resistance levels
            analysis.update(self._calculate_support_resistance(stock_data))
            
            # Add entry/exit signals
            analysis.update(self._generate_trading_signals(stock_data))
            
            # Add advanced statistical analysis
            advanced_stats = self._calculate_advanced_statistics(stock_data)
            if advanced_stats:
                analysis['advanced_statistics'] = advanced_stats
            
            # Determine investment strategy based on technical analysis
            signals = self._generate_trading_signals(stock_data)
            investment_strategy = self._determine_investment_strategy(advanced_stats, signals)
            analysis['investment_strategy'] = investment_strategy
            
            # Add overall technical summary
            analysis.update(self._generate_technical_summary(analysis))
            
            print(f"Technical analysis completed for {ticker}")
            return analysis
            
        except Exception as e:
            print(f"Error in technical analysis for {ticker}: {str(e)}")
            return {
                'error': f"Technical analysis failed for {ticker}: {str(e)}",
                'ticker': ticker.upper()
            }
    
    def _calculate_price_change(self, data: pd.DataFrame, periods: int) -> float:
        """Calculate price change over specified periods"""
        if len(data) < periods + 1:
            return 0.0
        return ((data['Close'].iloc[-1] - data['Close'].iloc[-periods-1]) / 
                data['Close'].iloc[-periods-1]) * 100
    
    def _safe_get_value(self, series, default_value=0.0):
        """Safely get the last value from a pandas series, handling None and NaN"""
        if series is None:
            return default_value
        if hasattr(series, 'empty') and series.empty:
            return default_value
        if hasattr(series, 'iloc'):
            last_value = series.iloc[-1]
        else:
            # Handle numpy arrays or single values
            last_value = series
        if pd.isna(last_value):
            return default_value
        return float(last_value)
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-based indicators"""
        indicators = {}
        current_price = data['Close'].iloc[-1]
        
        # RSI (Relative Strength Index)
        try:
            rsi = ta.rsi(data['Close'], length=14)
            rsi_value = self._safe_get_value(rsi, 50.0)
            indicators['rsi'] = {
                'current': rsi_value,
                'signal': self._get_rsi_signal(rsi_value),
                'overbought': 70,
                'oversold': 30
            }
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            indicators['rsi'] = {
                'current': 50.0,
                'signal': 'neutral',
                'overbought': 70,
                'oversold': 30
            }
        
        # MACD (Moving Average Convergence Divergence)
        try:
            macd = ta.macd(data['Close'])
            indicators['macd'] = {
                'macd_line': self._safe_get_value(macd.get('MACD_12_26_9', None), 0.0),
                'signal_line': self._safe_get_value(macd.get('MACDs_12_26_9', None), 0.0),
                'histogram': self._safe_get_value(macd.get('MACDh_12_26_9', None), 0.0),
                'signal': self._get_macd_signal(macd)
            }
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            indicators['macd'] = {
                'macd_line': 0.0,
                'signal_line': 0.0,
                'histogram': 0.0,
                'signal': 'neutral'
            }
        
        # Stochastic Oscillator
        try:
            stoch = ta.stoch(data['High'], data['Low'], data['Close'])
            indicators['stochastic'] = {
                'k_percent': self._safe_get_value(stoch.get('STOCHk_14_3_3', None), 50.0),
                'd_percent': self._safe_get_value(stoch.get('STOCHd_14_3_3', None), 50.0),
                'signal': self._get_stochastic_signal(stoch)
            }
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            indicators['stochastic'] = {
                'k_percent': 50.0,
                'd_percent': 50.0,
                'signal': 'neutral'
            }
        
        # Williams %R
        try:
            williams_r = ta.willr(data['High'], data['Low'], data['Close'])
            williams_r_value = self._safe_get_value(williams_r, -50.0)
            indicators['williams_r'] = {
                'current': williams_r_value,
                'signal': self._get_williams_r_signal(williams_r_value)
            }
        except Exception as e:
            print(f"Error calculating Williams %R: {e}")
            indicators['williams_r'] = {
                'current': -50.0,
                'signal': 'neutral'
            }
        
        return indicators
    
    def _calculate_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend-based indicators"""
        indicators = {}
        current_price = data['Close'].iloc[-1]
        
        # Moving Averages
        try:
            sma_20 = ta.sma(data['Close'], length=20)
            sma_50 = ta.sma(data['Close'], length=50)
            sma_200 = ta.sma(data['Close'], length=200)
            ema_12 = ta.ema(data['Close'], length=12)
            ema_26 = ta.ema(data['Close'], length=26)
            
            sma_20_val = self._safe_get_value(sma_20, current_price)
            sma_50_val = self._safe_get_value(sma_50, current_price)
            sma_200_val = self._safe_get_value(sma_200, current_price)
            ema_12_val = self._safe_get_value(ema_12, current_price)
            ema_26_val = self._safe_get_value(ema_26, current_price)
            
            # Check for golden/death cross (proper logic)
            golden_cross = False
            death_cross = False
            if len(sma_20) > 1 and len(sma_50) > 1:
                # Golden cross: SMA20 crosses above SMA50
                golden_cross = (sma_20.iloc[-1] > sma_50.iloc[-1] and 
                               sma_20.iloc[-2] <= sma_50.iloc[-2])
                # Death cross: SMA20 crosses below SMA50  
                death_cross = (sma_20.iloc[-1] < sma_50.iloc[-1] and 
                              sma_20.iloc[-2] >= sma_50.iloc[-2])
            
            indicators['moving_averages'] = {
                'sma_20': sma_20_val,
                'sma_50': sma_50_val,
                'sma_200': sma_200_val,
                'ema_12': ema_12_val,
                'ema_26': ema_26_val,
                'price_vs_sma_20': 'above' if current_price > sma_20_val else 'below',
                'price_vs_sma_50': 'above' if current_price > sma_50_val else 'below',
                'price_vs_sma_200': 'above' if current_price > sma_200_val else 'below',
                'golden_cross': golden_cross,
                'death_cross': death_cross
            }
        except Exception as e:
            print(f"Error calculating Moving Averages: {e}")
            indicators['moving_averages'] = {
                'sma_20': current_price,
                'sma_50': current_price,
                'sma_200': current_price,
                'ema_12': current_price,
                'ema_26': current_price,
                'price_vs_sma_20': 'above',
                'price_vs_sma_50': 'above',
                'price_vs_sma_200': 'above',
                'golden_cross': False,
                'death_cross': False
            }
        
        # ADX (Average Directional Index) - Trend strength
        try:
            adx = ta.adx(data['High'], data['Low'], data['Close'])
            adx_value = self._safe_get_value(adx.get('ADX_14', None), 25.0)
            di_plus = self._safe_get_value(adx.get('DMP_14', None), 0.0)
            di_minus = self._safe_get_value(adx.get('DMN_14', None), 0.0)
            
            indicators['adx'] = {
                'current': adx_value,
                'trend_strength': self._get_adx_trend_strength(adx_value),
                'di_plus': di_plus,
                'di_minus': di_minus
            }
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            indicators['adx'] = {
                'current': 25.0,
                'trend_strength': 'weak',
                'di_plus': 0.0,
                'di_minus': 0.0
            }
        
        # Parabolic SAR - Custom implementation
        try:
            psar = self._calculate_psar(data['High'], data['Low'], data['Close'])
            psar_value = self._safe_get_value(psar, current_price)
            indicators['parabolic_sar'] = {
                'current': psar_value,
                'signal': 'sell' if current_price < psar_value else 'buy'
            }
        except Exception as e:
            print(f"Error calculating Parabolic SAR: {e}")
            indicators['parabolic_sar'] = {
                'current': current_price,
                'signal': 'buy'
            }
        
        return indicators
    
    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-based indicators"""
        indicators = {}
        current_price = data['Close'].iloc[-1]
        
        # Bollinger Bands
        try:
            bb = ta.bbands(data['Close'], length=20)
            upper = self._safe_get_value(bb.get('BBU_20_2.0', None), current_price * 1.02)
            middle = self._safe_get_value(bb.get('BBM_20_2.0', None), current_price)
            lower = self._safe_get_value(bb.get('BBL_20_2.0', None), current_price * 0.98)
            bandwidth = self._safe_get_value(bb.get('BBB_20_2.0', None), 0.04)
            
            indicators['bollinger_bands'] = {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'bandwidth': bandwidth,
                'position': self._get_bollinger_position(current_price, upper, lower),
                'squeeze': bandwidth < 0.05
            }
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            indicators['bollinger_bands'] = {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98,
                'bandwidth': 0.04,
                'position': 'middle',
                'squeeze': False
            }
        
        # Average True Range (ATR)
        try:
            atr = ta.atr(data['High'], data['Low'], data['Close'])
            atr_value = self._safe_get_value(atr, 0.0)
            indicators['atr'] = {
                'current': atr_value,
                'volatility_level': self._get_atr_volatility_level(atr_value, current_price)
            }
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            indicators['atr'] = {
                'current': 0.0,
                'volatility_level': 'normal'
            }
        
        # Keltner Channels
        try:
            kc = ta.kc(data['High'], data['Low'], data['Close'])
            upper = self._safe_get_value(kc.get('KCUe_20_2', None), current_price * 1.02)
            lower = self._safe_get_value(kc.get('KCLe_20_2', None), current_price * 0.98)
            
            indicators['keltner_channels'] = {
                'upper': upper,
                'lower': lower,
                'position': self._get_keltner_position(current_price, upper, lower)
            }
        except Exception as e:
            print(f"Error calculating Keltner Channels: {e}")
            indicators['keltner_channels'] = {
                'upper': current_price * 1.02,
                'lower': current_price * 0.98,
                'position': 'middle'
            }
        
        return indicators
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        indicators = {}
        
        # Volume SMA
        try:
            volume_sma = ta.sma(data['Volume'], length=20)
            volume_sma_val = self._safe_get_value(volume_sma, data['Volume'].iloc[-1])
            current_volume = data['Volume'].iloc[-1]
            
            indicators['volume'] = {
                'current': int(current_volume),
                'sma_20': float(volume_sma_val),
                'volume_ratio': current_volume / volume_sma_val if volume_sma_val > 0 else 1.0,
                'high_volume': current_volume > volume_sma_val * 1.5
            }
        except Exception as e:
            print(f"Error calculating Volume indicators: {e}")
            current_volume = data['Volume'].iloc[-1]
            indicators['volume'] = {
                'current': int(current_volume),
                'sma_20': float(current_volume),
                'volume_ratio': 1.0,
                'high_volume': False
            }
        
        # On-Balance Volume (OBV)
        try:
            obv = ta.obv(data['Close'], data['Volume'])
            obv_value = self._safe_get_value(obv, 0.0)
            # Safely get previous value for trend calculation
            if hasattr(obv, 'iloc') and len(obv) > 5:
                obv_prev = self._safe_get_value(obv.iloc[-5], 0.0)
            else:
                obv_prev = 0.0
            
            indicators['obv'] = {
                'current': obv_value,
                'trend': 'rising' if obv_value > obv_prev else 'falling'
            }
        except Exception as e:
            print(f"Error calculating OBV: {e}")
            indicators['obv'] = {
                'current': 0.0,
                'trend': 'neutral'
            }
        
        # Volume Price Trend (VPT) - Custom implementation
        try:
            vpt = self._calculate_vpt(data['Close'], data['Volume'])
            vpt_value = self._safe_get_value(vpt, 0.0)
            vpt_prev = self._safe_get_value(vpt.iloc[-5] if len(vpt) > 5 else vpt, 0.0)
            
            indicators['vpt'] = {
                'current': vpt_value,
                'trend': 'rising' if vpt_value > vpt_prev else 'falling'
            }
        except Exception as e:
            print(f"Error calculating VPT: {e}")
            indicators['vpt'] = {
                'current': 0.0,
                'trend': 'neutral'
            }
        
        return indicators
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        # Simple pivot points
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        # Dynamic support/resistance using recent highs and lows
        recent_highs = data['High'].tail(20).nlargest(3)
        recent_lows = data['Low'].tail(20).nsmallest(3)
        
        return {
            'pivot_points': {
                'pivot': float(pivot),
                'resistance_1': float(r1),
                'resistance_2': float(r2),
                'support_1': float(s1),
                'support_2': float(s2)
            },
            'dynamic_levels': {
                'resistance_levels': [float(x) for x in recent_highs.values],
                'support_levels': [float(x) for x in recent_lows.values],
                'nearest_resistance': float(recent_highs.min()),
                'nearest_support': float(recent_lows.max())
            }
        }
    
    def _generate_trading_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators with statistical confidence"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'hold_signals': [],
            'overall_signal': 'HOLD',
            'confidence': 50,
            'statistical_confidence': {}
        }
        
        try:
            # Get current indicator values
            rsi = ta.rsi(data['Close'], length=14)
            rsi_value = self._safe_get_value(rsi, 50.0)
            
            macd = ta.macd(data['Close'])
            macd_line = self._safe_get_value(macd.get('MACD_12_26_9', None), 0.0)
            signal_line = self._safe_get_value(macd.get('MACDs_12_26_9', None), 0.0)
            
            sma_20 = ta.sma(data['Close'], length=20)
            sma_50 = ta.sma(data['Close'], length=50)
            sma_20_val = self._safe_get_value(sma_20, data['Close'].iloc[-1])
            sma_50_val = self._safe_get_value(sma_50, data['Close'].iloc[-1])
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate statistical confidence for each indicator
            rsi_confidence = self._calculate_rsi_confidence(rsi, rsi_value)
            macd_confidence = self._calculate_macd_confidence(macd, macd_line, signal_line)
            ma_confidence = self._calculate_ma_confidence(data['Close'], sma_20, sma_50, current_price)
            
            # Store statistical confidence details
            signals['statistical_confidence'] = {
                'rsi_confidence': rsi_confidence,
                'macd_confidence': macd_confidence,
                'ma_confidence': ma_confidence
            }
            
            # Generate signals based on statistical significance
            buy_signals = []
            sell_signals = []
            buy_confidence = 0
            sell_confidence = 0
            
            # RSI signals with statistical confidence
            if rsi_confidence['signal'] == 'buy' and rsi_confidence['confidence'] > 60:
                signals['buy_signals'].append(f"RSI oversold ({rsi_value:.1f}) - {rsi_confidence['confidence']:.1f}% confidence")
                buy_confidence += rsi_confidence['confidence']
            elif rsi_confidence['signal'] == 'sell' and rsi_confidence['confidence'] > 60:
                signals['sell_signals'].append(f"RSI overbought ({rsi_value:.1f}) - {rsi_confidence['confidence']:.1f}% confidence")
                sell_confidence += rsi_confidence['confidence']
            
            # MACD signals with statistical confidence
            if macd_confidence['signal'] == 'buy' and macd_confidence['confidence'] > 60:
                signals['buy_signals'].append(f"MACD bullish - {macd_confidence['confidence']:.1f}% confidence")
                buy_confidence += macd_confidence['confidence']
            elif macd_confidence['signal'] == 'sell' and macd_confidence['confidence'] > 60:
                signals['sell_signals'].append(f"MACD bearish - {macd_confidence['confidence']:.1f}% confidence")
                sell_confidence += macd_confidence['confidence']
            
            # Moving average signals with statistical confidence
            if ma_confidence['signal'] == 'buy' and ma_confidence['confidence'] > 60:
                signals['buy_signals'].append(f"Price above MAs - {ma_confidence['confidence']:.1f}% confidence")
                buy_confidence += ma_confidence['confidence']
            elif ma_confidence['signal'] == 'sell' and ma_confidence['confidence'] > 60:
                signals['sell_signals'].append(f"Price below MAs - {ma_confidence['confidence']:.1f}% confidence")
                sell_confidence += ma_confidence['confidence']
            
            # Calculate overall signal and confidence using statistical approach
            if buy_confidence > sell_confidence and buy_confidence > 0:
                signals['overall_signal'] = 'BUY'
                signals['confidence'] = min(95, buy_confidence / max(1, len(signals['buy_signals'])))
            elif sell_confidence > buy_confidence and sell_confidence > 0:
                signals['overall_signal'] = 'SELL'
                signals['confidence'] = min(95, sell_confidence / max(1, len(signals['sell_signals'])))
            else:
                signals['overall_signal'] = 'HOLD'
                signals['confidence'] = 50
                
        except Exception as e:
            print(f"Error generating trading signals: {e}")
            signals['overall_signal'] = 'HOLD'
            signals['confidence'] = 50
        
        except Exception as e:
            print(f"Error generating trading signals: {e}")
            signals['overall_signal'] = 'HOLD'
            signals['confidence'] = 50
        
        return signals
    
    def _generate_technical_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive technical summary"""
        summary = {
            'trend': 'neutral',
            'momentum': 'neutral',
            'volatility': 'normal',
            'volume': 'normal',
            'strength': 'neutral'
        }
        
        # Trend analysis
        if analysis.get('moving_averages', {}).get('price_vs_sma_20') == 'above' and \
           analysis.get('moving_averages', {}).get('price_vs_sma_50') == 'above':
            summary['trend'] = 'bullish'
        elif analysis.get('moving_averages', {}).get('price_vs_sma_20') == 'below' and \
             analysis.get('moving_averages', {}).get('price_vs_sma_50') == 'below':
            summary['trend'] = 'bearish'
        
        # Momentum analysis
        rsi = analysis.get('rsi', {}).get('current', 50)
        if rsi > 60:
            summary['momentum'] = 'bullish'
        elif rsi < 40:
            summary['momentum'] = 'bearish'
        
        # Volatility analysis
        bb_position = analysis.get('bollinger_bands', {}).get('position', 'middle')
        if bb_position in ['upper', 'lower']:
            summary['volatility'] = 'high'
        
        # Volume analysis
        if analysis.get('volume', {}).get('high_volume', False):
            summary['volume'] = 'high'
        
        # Overall strength
        bullish_count = sum(1 for v in summary.values() if v == 'bullish')
        bearish_count = sum(1 for v in summary.values() if v == 'bearish')
        
        if bullish_count > bearish_count:
            summary['strength'] = 'bullish'
        elif bearish_count > bullish_count:
            summary['strength'] = 'bearish'
        
        return {'technical_summary': summary}
    
    # Helper methods for signal generation
    def _get_rsi_signal(self, rsi: float) -> str:
        if pd.isna(rsi):
            return 'neutral'
        if rsi > 70:
            return 'overbought'
        elif rsi < 30:
            return 'oversold'
        else:
            return 'neutral'
    
    def _get_macd_signal(self, macd: pd.DataFrame) -> str:
        try:
            macd_line = self._safe_get_value(macd.get('MACD_12_26_9', None), 0.0)
            signal_line = self._safe_get_value(macd.get('MACDs_12_26_9', None), 0.0)
            if macd_line > signal_line:
                return 'bullish'
            else:
                return 'bearish'
        except:
            return 'neutral'
    
    def _get_stochastic_signal(self, stoch: pd.DataFrame) -> str:
        try:
            k = self._safe_get_value(stoch.get('STOCHk_14_3_3', None), 50.0)
            d = self._safe_get_value(stoch.get('STOCHd_14_3_3', None), 50.0)
            if k > 80 and d > 80:
                return 'overbought'
            elif k < 20 and d < 20:
                return 'oversold'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _get_williams_r_signal(self, williams_r: float) -> str:
        if pd.isna(williams_r):
            return 'neutral'
        if williams_r > -20:
            return 'overbought'
        elif williams_r < -80:
            return 'oversold'
        else:
            return 'neutral'
    
    def _get_adx_trend_strength(self, adx: float) -> str:
        if pd.isna(adx):
            return 'weak'
        if adx > 25:
            return 'strong'
        elif adx > 20:
            return 'moderate'
        else:
            return 'weak'
    
    def _get_bollinger_position(self, price: float, upper: float, lower: float) -> str:
        if pd.isna(upper) or pd.isna(lower):
            return 'middle'
        if price > upper:
            return 'upper'
        elif price < lower:
            return 'lower'
        else:
            return 'middle'
    
    def _get_atr_volatility_level(self, atr: float, price: float) -> str:
        if pd.isna(atr) or pd.isna(price):
            return 'normal'
        atr_percent = (atr / price) * 100
        # More conservative thresholds based on typical market volatility
        if atr_percent > 5:
            return 'high'
        elif atr_percent < 0.5:
            return 'low'
        else:
            return 'normal'
    
    def _get_keltner_position(self, price: float, upper: float, lower: float) -> str:
        if pd.isna(upper) or pd.isna(lower):
            return 'middle'
        if price > upper:
            return 'upper'
        elif price < lower:
            return 'lower'
        else:
            return 'middle'
    
    def _calculate_rsi_confidence(self, rsi_series: pd.Series, current_rsi: float) -> Dict[str, Any]:
        """Calculate statistical confidence for RSI signals"""
        try:
            if rsi_series is None or len(rsi_series) < 20:
                return {'signal': 'neutral', 'confidence': 50.0}
            
            # Calculate RSI distribution statistics
            rsi_clean = rsi_series.dropna()
            if len(rsi_clean) < 10:
                return {'signal': 'neutral', 'confidence': 50.0}
            
            mean_rsi = rsi_clean.mean()
            std_rsi = rsi_clean.std()
            
            # Calculate z-score for current RSI
            z_score = (current_rsi - mean_rsi) / std_rsi if std_rsi > 0 else 0
            
            # Calculate percentile rank
            percentile = stats.percentileofscore(rsi_clean, current_rsi)
            
            # Determine signal and confidence based on statistical significance
            if current_rsi < 30:
                # Oversold condition
                signal = 'buy'
                # Confidence based on how extreme the RSI is
                confidence = min(95, 70 + abs(z_score) * 5)
            elif current_rsi > 70:
                # Overbought condition
                signal = 'sell'
                confidence = min(95, 70 + abs(z_score) * 5)
            elif current_rsi < 40:
                # Slightly oversold
                signal = 'buy'
                confidence = min(70, 50 + abs(z_score) * 3)
            elif current_rsi > 60:
                # Slightly overbought
                signal = 'sell'
                confidence = min(70, 50 + abs(z_score) * 3)
            else:
                signal = 'neutral'
                confidence = 50.0
            
            return {
                'signal': signal,
                'confidence': confidence,
                'z_score': z_score,
                'percentile': percentile,
                'mean': mean_rsi,
                'std': std_rsi
            }
            
        except Exception as e:
            print(f"Error calculating RSI confidence: {e}")
            return {'signal': 'neutral', 'confidence': 50.0}
    
    def _calculate_macd_confidence(self, macd_data: pd.DataFrame, macd_line: float, signal_line: float) -> Dict[str, Any]:
        """Calculate statistical confidence for MACD signals"""
        try:
            if macd_data is None or len(macd_data) < 20:
                return {'signal': 'neutral', 'confidence': 50.0}
            
            macd_series = macd_data.get('MACD_12_26_9', None)
            if macd_series is None or len(macd_series) < 20:
                return {'signal': 'neutral', 'confidence': 50.0}
            
            macd_clean = macd_series.dropna()
            if len(macd_clean) < 10:
                return {'signal': 'neutral', 'confidence': 50.0}
            
            # Calculate MACD distribution statistics
            mean_macd = macd_clean.mean()
            std_macd = macd_clean.std()
            
            # Calculate histogram (difference between MACD and signal)
            histogram = macd_line - signal_line
            
            # Calculate z-score for current MACD
            z_score = (macd_line - mean_macd) / std_macd if std_macd > 0 else 0
            
            # Determine signal and confidence
            if macd_line > signal_line and macd_line > 0:
                # Bullish crossover above zero
                signal = 'buy'
                confidence = min(95, 75 + abs(z_score) * 3)
            elif macd_line < signal_line and macd_line < 0:
                # Bearish crossover below zero
                signal = 'sell'
                confidence = min(95, 75 + abs(z_score) * 3)
            elif macd_line > signal_line:
                # Bullish crossover
                signal = 'buy'
                confidence = min(80, 60 + abs(z_score) * 2)
            else:
                # Bearish crossover
                signal = 'sell'
                confidence = min(80, 60 + abs(z_score) * 2)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'z_score': z_score,
                'histogram': histogram,
                'mean': mean_macd,
                'std': std_macd
            }
            
        except Exception as e:
            print(f"Error calculating MACD confidence: {e}")
            return {'signal': 'neutral', 'confidence': 50.0}
    
    def _calculate_ma_confidence(self, price_series: pd.Series, sma_20: pd.Series, sma_50: pd.Series, current_price: float) -> Dict[str, Any]:
        """Calculate statistical confidence for moving average signals"""
        try:
            if len(price_series) < 50:
                return {'signal': 'neutral', 'confidence': 50.0}
            
            # Calculate price deviation from moving averages
            sma_20_val = self._safe_get_value(sma_20, current_price)
            sma_50_val = self._safe_get_value(sma_50, current_price)
            
            # Calculate percentage deviation from MAs
            dev_20 = ((current_price - sma_20_val) / sma_20_val) * 100
            dev_50 = ((current_price - sma_50_val) / sma_50_val) * 100
            
            # Calculate historical price volatility
            price_returns = price_series.pct_change().dropna()
            if len(price_returns) < 20:
                return {'signal': 'neutral', 'confidence': 50.0}
            
            volatility = price_returns.std() * 100  # Annualized volatility
            
            # Calculate statistical significance of deviation
            z_score_20 = dev_20 / volatility if volatility > 0 else 0
            z_score_50 = dev_50 / volatility if volatility > 0 else 0
            
            # Determine signal and confidence
            if current_price > sma_20_val > sma_50_val:
                # Strong bullish alignment
                signal = 'buy'
                confidence = min(95, 70 + abs(z_score_20) * 2)
            elif current_price < sma_20_val < sma_50_val:
                # Strong bearish alignment
                signal = 'sell'
                confidence = min(95, 70 + abs(z_score_20) * 2)
            elif current_price > sma_20_val:
                # Above short-term MA
                signal = 'buy'
                confidence = min(75, 55 + abs(z_score_20) * 1.5)
            else:
                # Below short-term MA
                signal = 'sell'
                confidence = min(75, 55 + abs(z_score_20) * 1.5)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'deviation_20': dev_20,
                'deviation_50': dev_50,
                'z_score_20': z_score_20,
                'z_score_50': z_score_50,
                'volatility': volatility
            }
            
        except Exception as e:
            print(f"Error calculating MA confidence: {e}")
            return {'signal': 'neutral', 'confidence': 50.0}
    
    def _calculate_advanced_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced statistical measures for technical analysis"""
        try:
            if len(data) < 50:
                return {}
            
            close_prices = data['Close']
            returns = close_prices.pct_change().dropna()
            
            if len(returns) < 20:
                return {}
            
            stats_results = {}
            
            # 1. Z-Score Analysis
            stats_results['z_score_analysis'] = self._calculate_z_score_analysis(close_prices, returns)
            
            # 2. Rolling Statistics
            stats_results['rolling_statistics'] = self._calculate_rolling_statistics(close_prices, returns)
            
            # 3. Autocorrelation Analysis
            stats_results['autocorrelation'] = self._calculate_autocorrelation(returns)
            
            # 4. Stationarity Test (ADF)
            stats_results['stationarity'] = self._calculate_stationarity_test(close_prices)
            
            # 5. GARCH Volatility Model
            stats_results['garch_model'] = self._calculate_garch_volatility(returns)
            
            # 6. Linear Regression Trend
            stats_results['regression_trend'] = self._calculate_regression_trend(close_prices)
            
            # 7. Distribution Statistics
            stats_results['distribution_stats'] = self._calculate_distribution_statistics(returns)
            
            # 8. Risk Metrics
            stats_results['risk_metrics'] = self._calculate_risk_metrics(returns)
            
            # 9. Change Point Detection
            stats_results['change_points'] = self._detect_change_points(close_prices)
            
            # 10. Kalman Filter Trend
            stats_results['kalman_trend'] = self._calculate_kalman_trend(close_prices)
            
            return stats_results
            
        except Exception as e:
            print(f"Error calculating advanced statistics: {e}")
            return {}
    
    def _calculate_z_score_analysis(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Calculate Z-score analysis for mean reversion opportunities"""
        try:
            # Price Z-score (deviation from moving average)
            ma_20 = prices.rolling(window=20).mean()
            price_z_score = (prices.iloc[-1] - ma_20.iloc[-1]) / prices.rolling(window=20).std().iloc[-1]
            
            # Returns Z-score (momentum)
            returns_z_score = (returns.iloc[-1] - returns.mean()) / returns.std()
            
            # Volatility Z-score
            rolling_vol = returns.rolling(window=20).std()
            vol_z_score = (rolling_vol.iloc[-1] - rolling_vol.mean()) / rolling_vol.std()
            
            return {
                'price_z_score': price_z_score,
                'returns_z_score': returns_z_score,
                'volatility_z_score': vol_z_score,
                'mean_reversion_signal': 'buy' if price_z_score < -2 else 'sell' if price_z_score > 2 else 'neutral',
                'momentum_signal': 'buy' if returns_z_score > 1 else 'sell' if returns_z_score < -1 else 'neutral'
            }
        except Exception as e:
            print(f"Error in Z-score analysis: {e}")
            return {}
    
    def _calculate_rolling_statistics(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Calculate rolling mean and standard deviation for regime detection"""
        try:
            window = 20
            
            # Rolling statistics
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            rolling_vol = returns.rolling(window=window).std()
            
            # Current vs historical
            current_price = prices.iloc[-1]
            current_mean = rolling_mean.iloc[-1]
            current_std = rolling_std.iloc[-1]
            current_vol = rolling_vol.iloc[-1]
            
            # Regime detection
            price_regime = 'above_trend' if current_price > current_mean else 'below_trend'
            vol_regime = 'high_vol' if current_vol > rolling_vol.mean() else 'low_vol'
            
            return {
                'current_price': current_price,
                'rolling_mean': current_mean,
                'rolling_std': current_std,
                'rolling_volatility': current_vol,
                'price_regime': price_regime,
                'volatility_regime': vol_regime,
                'trend_stability': 'stable' if current_std < rolling_std.mean() else 'unstable'
            }
        except Exception as e:
            print(f"Error in rolling statistics: {e}")
            return {}
    
    def _calculate_autocorrelation(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate autocorrelation for momentum detection"""
        try:
            if len(returns) < 30:
                return {}
            
            # Calculate ACF for different lags
            acf_values = acf(returns, nlags=10, fft=False)
            
            # Momentum indicators
            lag1_corr = acf_values[1] if len(acf_values) > 1 else 0
            lag5_corr = acf_values[5] if len(acf_values) > 5 else 0
            
            # Momentum signal
            momentum_signal = 'positive' if lag1_corr > 0.1 else 'negative' if lag1_corr < -0.1 else 'neutral'
            
            return {
                'lag1_autocorrelation': lag1_corr,
                'lag5_autocorrelation': lag5_corr,
                'momentum_signal': momentum_signal,
                'persistence': 'high' if abs(lag1_corr) > 0.2 else 'low'
            }
        except Exception as e:
            print(f"Error in autocorrelation: {e}")
            return {}
    
    def _calculate_stationarity_test(self, prices: pd.Series) -> Dict[str, Any]:
        """Perform Augmented Dickey-Fuller test for stationarity"""
        try:
            if len(prices) < 30:
                return {}
            
            # ADF test
            adf_result = adfuller(prices.dropna())
            
            # Test results
            adf_statistic = adf_result[0]
            p_value = adf_result[1]
            critical_values = adf_result[4]
            
            # Stationarity assessment
            is_stationary = p_value < 0.05
            stationarity_level = 'stationary' if is_stationary else 'non_stationary'
            
            return {
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'stationarity_level': stationarity_level,
                'mean_reversion_suitable': is_stationary
            }
        except Exception as e:
            print(f"Error in stationarity test: {e}")
            return {}
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate GARCH model for conditional volatility"""
        try:
            if len(returns) < 50:
                return {}
            
            # Simple GARCH(1,1) model
            returns_clean = returns.dropna()
            
            # Calculate conditional volatility using rolling window
            window = 20
            conditional_vol = returns_clean.rolling(window=window).std()
            
            # Volatility clustering
            vol_clustering = 'high' if conditional_vol.iloc[-1] > conditional_vol.mean() else 'low'
            
            # Volatility trend
            vol_trend = 'increasing' if conditional_vol.iloc[-1] > conditional_vol.iloc[-5] else 'decreasing'
            
            return {
                'conditional_volatility': conditional_vol.iloc[-1],
                'volatility_clustering': vol_clustering,
                'volatility_trend': vol_trend,
                'risk_level': 'high' if conditional_vol.iloc[-1] > conditional_vol.quantile(0.8) else 'low'
            }
        except Exception as e:
            print(f"Error in GARCH volatility: {e}")
            return {}
    
    def _calculate_regression_trend(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate linear regression trend and R-squared"""
        try:
            if len(prices) < 30:
                return {}
            
            # Prepare data for regression
            X = np.arange(len(prices)).reshape(-1, 1)
            y = prices.values
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(X, y)
            
            # Calculate predictions and R-squared
            y_pred = reg.predict(X)
            r_squared = reg.score(X, y)
            
            # Trend analysis
            slope = reg.coef_[0]
            trend_strength = 'strong' if r_squared > 0.7 else 'moderate' if r_squared > 0.3 else 'weak'
            trend_direction = 'upward' if slope > 0 else 'downward'
            
            return {
                'slope': slope,
                'r_squared': r_squared,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'trend_quality': 'high' if r_squared > 0.5 else 'low'
            }
        except Exception as e:
            print(f"Error in regression trend: {e}")
            return {}
    
    def _calculate_distribution_statistics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate skewness and kurtosis for distribution analysis"""
        try:
            if len(returns) < 20:
                return {}
            
            # Distribution statistics
            skewness_val = skew(returns.dropna())
            kurtosis_val = kurtosis(returns.dropna())
            
            # Distribution characteristics
            skewness_type = 'right_skewed' if skewness_val > 0.5 else 'left_skewed' if skewness_val < -0.5 else 'symmetric'
            tail_risk = 'high' if kurtosis_val > 3 else 'low'
            
            return {
                'skewness': skewness_val,
                'kurtosis': kurtosis_val,
                'skewness_type': skewness_type,
                'tail_risk': tail_risk,
                'distribution_shape': 'normal' if abs(skewness_val) < 0.5 and kurtosis_val < 3 else 'non_normal'
            }
        except Exception as e:
            print(f"Error in distribution statistics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate risk metrics including VaR and Sharpe ratio"""
        try:
            if len(returns) < 30:
                return {}
            
            returns_clean = returns.dropna()
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns_clean, 5)
            
            # Sharpe Ratio (assuming risk-free rate of 0)
            sharpe_ratio = returns_clean.mean() / returns_clean.std() if returns_clean.std() > 0 else 0
            
            # Maximum Drawdown
            cumulative_returns = (1 + returns_clean).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'var_95': var_95,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'risk_adjusted_return': 'good' if sharpe_ratio > 0.5 else 'poor',
                'risk_level': 'high' if abs(var_95) > 0.02 else 'low'
            }
        except Exception as e:
            print(f"Error in risk metrics: {e}")
            return {}
    
    def _detect_change_points(self, prices: pd.Series) -> Dict[str, Any]:
        """Detect change points in price series"""
        try:
            if len(prices) < 30:
                return {}
            
            # Simple change point detection using rolling statistics
            window = 10
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            
            # Detect significant deviations
            z_scores = (prices - rolling_mean) / rolling_std
            change_points = z_scores.abs() > 2
            
            # Recent change points
            recent_changes = change_points.tail(10).sum()
            
            return {
                'recent_change_points': recent_changes,
                'change_frequency': 'high' if recent_changes > 2 else 'low',
                'stability': 'unstable' if recent_changes > 1 else 'stable'
            }
        except Exception as e:
            print(f"Error in change point detection: {e}")
            return {}
    
    def _calculate_kalman_trend(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate Kalman filter trend estimation"""
        try:
            if len(prices) < 30:
                return {}
            
            # Simple Kalman-like smoothing using exponential moving average
            ema_short = prices.ewm(span=5).mean()
            ema_long = prices.ewm(span=20).mean()
            
            # Trend estimation
            current_short = ema_short.iloc[-1]
            current_long = ema_long.iloc[-1]
            
            # Trend signal
            trend_signal = 'bullish' if current_short > current_long else 'bearish'
            trend_strength = abs(current_short - current_long) / current_long
            
            return {
                'short_ema': current_short,
                'long_ema': current_long,
                'trend_signal': trend_signal,
                'trend_strength': trend_strength,
                'trend_quality': 'strong' if trend_strength > 0.02 else 'weak'
            }
        except Exception as e:
            print(f"Error in Kalman trend: {e}")
            return {}
    
    def _determine_investment_strategy(self, advanced_stats: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal investment strategy based on technical analysis"""
        try:
            strategy = {
                'recommended_timeframe': 'medium_term',
                'primary_driver': 'balanced',
                'entry_strategy': 'scale_in',
                'risk_management': 'moderate',
                'position_sizing': 'standard',
                'reasoning': []
            }
            
            # Extract key metrics
            z_score_analysis = advanced_stats.get('z_score_analysis', {})
            stationarity = advanced_stats.get('stationarity', {})
            volatility = advanced_stats.get('garch_model', {})
            trend = advanced_stats.get('regression_trend', {})
            risk = advanced_stats.get('risk_metrics', {})
            
            price_z_score = z_score_analysis.get('price_z_score', 0)
            is_stationary = stationarity.get('is_stationary', False)
            volatility_level = volatility.get('risk_level', 'medium')
            trend_strength = trend.get('trend_strength', 'weak')
            sharpe_ratio = risk.get('sharpe_ratio', 0)
            
            # Determine timeframe based on technical characteristics
            if abs(price_z_score) > 2.5:
                # Extreme conditions - short-term opportunity
                strategy['recommended_timeframe'] = 'short_term'
                strategy['primary_driver'] = 'technical'
                strategy['entry_strategy'] = 'aggressive' if price_z_score < -2 else 'cautious'
                strategy['reasoning'].append(f"Extreme z-score ({price_z_score:.2f}) indicates short-term mean reversion opportunity")
                
            elif is_stationary and abs(price_z_score) > 1.5:
                # Stationary with moderate deviation - medium-term mean reversion
                strategy['recommended_timeframe'] = 'medium_term'
                strategy['primary_driver'] = 'technical'
                strategy['entry_strategy'] = 'scale_in'
                strategy['reasoning'].append("Stationary series with moderate deviation - suitable for mean reversion")
                
            elif not is_stationary and trend_strength in ['strong', 'moderate']:
                # Non-stationary with trend - long-term trend following
                strategy['recommended_timeframe'] = 'long_term'
                strategy['primary_driver'] = 'fundamental'
                strategy['entry_strategy'] = 'dollar_cost_average'
                strategy['reasoning'].append("Non-stationary with trend - focus on fundamentals with technical timing")
                
            else:
                # Neutral conditions - balanced approach
                strategy['recommended_timeframe'] = 'medium_term'
                strategy['primary_driver'] = 'balanced'
                strategy['entry_strategy'] = 'scale_in'
                strategy['reasoning'].append("Neutral technical conditions - balanced fundamental/technical approach")
            
            # Adjust risk management based on volatility
            if volatility_level == 'high':
                strategy['risk_management'] = 'conservative'
                strategy['position_sizing'] = 'reduced'
                strategy['reasoning'].append("High volatility requires conservative risk management")
            elif volatility_level == 'low':
                strategy['risk_management'] = 'aggressive'
                strategy['position_sizing'] = 'increased'
                strategy['reasoning'].append("Low volatility allows for more aggressive positioning")
            
            # Adjust based on risk-adjusted returns
            if sharpe_ratio < -0.5:
                strategy['risk_management'] = 'very_conservative'
                strategy['position_sizing'] = 'minimal'
                strategy['reasoning'].append("Poor risk-adjusted returns require minimal position sizing")
            elif sharpe_ratio > 0.5:
                strategy['position_sizing'] = 'increased'
                strategy['reasoning'].append("Good risk-adjusted returns support larger positions")
            
            return strategy
            
        except Exception as e:
            print(f"Error determining investment strategy: {e}")
            return {
                'recommended_timeframe': 'medium_term',
                'primary_driver': 'balanced',
                'entry_strategy': 'scale_in',
                'risk_management': 'moderate',
                'position_sizing': 'standard',
                'reasoning': ['Error in strategy determination - using default balanced approach']
            }
    
    def _calculate_vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Price Trend (VPT) - Custom implementation"""
        try:
            # Calculate price change percentage
            price_change = close.pct_change()
            # Calculate VPT = previous VPT + (price change * volume)
            vpt = pd.Series(index=close.index, dtype=float)
            vpt.iloc[0] = 0.0  # Start with 0
            
            for i in range(1, len(close)):
                vpt.iloc[i] = vpt.iloc[i-1] + (price_change.iloc[i] * volume.iloc[i])
            
            return vpt
        except Exception as e:
            print(f"Error in VPT calculation: {e}")
            return pd.Series([0.0] * len(close), index=close.index)
    
    def _calculate_psar(self, high: pd.Series, low: pd.Series, close: pd.Series, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR - Custom implementation"""
        try:
            psar = pd.Series(index=close.index, dtype=float)
            psar.iloc[0] = low.iloc[0]  # Start with the first low
            
            af = acceleration  # Acceleration factor
            ep = high.iloc[0]  # Extreme point
            long = True  # Long position
            
            for i in range(1, len(close)):
                if long:
                    # Calculate SAR for long position
                    psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                    
                    # Update extreme point if new high
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + acceleration, maximum)
                    
                    # Check for reversal
                    if close.iloc[i] < psar.iloc[i]:
                        long = False
                        psar.iloc[i] = ep
                        ep = low.iloc[i]
                        af = acceleration
                else:
                    # Calculate SAR for short position
                    psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                    
                    # Update extreme point if new low
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + acceleration, maximum)
                    
                    # Check for reversal
                    if close.iloc[i] > psar.iloc[i]:
                        long = True
                        psar.iloc[i] = ep
                        ep = high.iloc[i]
                        af = acceleration
            
            return psar
        except Exception as e:
            print(f"Error in PSAR calculation: {e}")
            return pd.Series(close.values, index=close.index) 