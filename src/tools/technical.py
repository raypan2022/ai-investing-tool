import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class TechnicalAnalyzer:
    """
    Minimal, high-signal technical analysis for short-term forecasting.

    Returned features focus on: trend, momentum, volatility, volume, and
    nearest support/resistance, plus a baseline synthesis (verdict + confidence).
    """

    def __init__(self, config):
        self.config = config

    def analyze_stock(self, ticker: str, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Run minimal technical analysis on OHLCV DataFrame (expects columns: Open, High, Low, Close, Volume)."""
        try:
            if stock_data.empty or len(stock_data) < 50:
                return {
                    'error': f"Insufficient data for {ticker}. Need at least 50 rows of OHLCV data.",
                    'ticker': ticker.upper()
                }

            close = stock_data['Close']
            high = stock_data['High']
            low = stock_data['Low']
            volume = stock_data['Volume']
            current_price = float(close.iloc[-1])

            # Basic returns
            price_change_1d = self._pct_change(close, 1)
            price_change_5d = self._pct_change(close, 5)
            price_change_20d = self._pct_change(close, 20)

            # Trend (SMAs)
            sma_20 = ta.sma(close, length=20)
            sma_50 = ta.sma(close, length=50)
            sma_200 = ta.sma(close, length=200)
            sma20_val = float(sma_20.iloc[-1]) if not sma_20.empty else current_price
            sma50_val = float(sma_50.iloc[-1]) if not sma_50.empty else current_price
            sma200_val = float(sma_200.iloc[-1]) if not sma_200.empty else current_price
            price_vs_sma_20 = 'above' if current_price > sma20_val else 'below'
            price_vs_sma_50 = 'above' if current_price > sma50_val else 'below'
            price_vs_sma_200 = 'above' if current_price > sma200_val else 'below'
            golden_cross = self._crossed_above(sma_20, sma_50)
            death_cross = self._crossed_below(sma_20, sma_50)

            # Momentum (RSI, MACD)
            rsi = ta.rsi(close, length=14)
            rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50.0
            rsi_zone = 'oversold' if rsi_val < 30 else 'overbought' if rsi_val > 70 else 'neutral'

            macd = ta.macd(close)
            macd_hist = float(macd['MACDh_12_26_9'].iloc[-1]) if macd is not None and 'MACDh_12_26_9' in macd.columns else 0.0
            macd_signal = 'bullish' if macd_hist > 0 else 'bearish' if macd_hist < 0 else 'neutral'

            # Volatility (Bollinger Bands + ATR)
            bb = ta.bbands(close, length=20, std=2)
            if bb is not None and {'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0'}.issubset(set(bb.columns)):
                bb_upper = float(bb['BBU_20_2.0'].iloc[-1])
                bb_middle = float(bb['BBM_20_2.0'].iloc[-1])
                bb_lower = float(bb['BBL_20_2.0'].iloc[-1])
                bb_position = 'upper' if current_price > bb_upper else 'lower' if current_price < bb_lower else 'middle'
                bb_bandwidth = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0.0
                bb_squeeze = bb_bandwidth < 0.05
            else:
                bb_position = 'middle'
                bb_squeeze = False

            atr = ta.atr(high, low, close)
            atr_val = float(atr.iloc[-1]) if not atr.empty else 0.0
            atr_pct = atr_val / current_price if current_price > 0 else 0.0

            # Volume
            vol_sma20 = volume.rolling(20).mean()
            vol_sma20_val = float(vol_sma20.iloc[-1]) if not vol_sma20.empty else float(volume.iloc[-1])
            volume_ratio = float(volume.iloc[-1]) / vol_sma20_val if vol_sma20_val > 0 else 1.0
            high_volume = volume_ratio > 1.5

            # Support/Resistance (simple pivots + recent extremes)
            pivots = self._pivot_points(high, low, close)
            nearest_support, nearest_resistance = self._nearest_levels(high, low, close)

            # Baseline synthesis (rule-based)
            buy_signals, sell_signals = self._collect_signals(
                rsi_val,
                rsi_zone,
                macd_hist,
                price_vs_sma_50,
                bb_position,
                volume_ratio,
                golden_cross,
                death_cross,
            )
            overall_signal, confidence = self._baseline_verdict(buy_signals, sell_signals)

            result: Dict[str, Any] = {
                'ticker': ticker.upper(),
                'analysis_date': datetime.now().isoformat(),
                'data_points': int(len(stock_data)),
                'current_price': current_price,
                # Returns
                'returns': {
                    'r1d': price_change_1d,
                    'r5d': price_change_5d,
                    'r20d': price_change_20d,
                },
                # Trend
                'moving_averages': {
                    'sma_20': sma20_val,
                    'sma_50': sma50_val,
                    'sma_200': sma200_val,
                    'price_vs_sma_20': price_vs_sma_20,
                    'price_vs_sma_50': price_vs_sma_50,
                    'price_vs_sma_200': price_vs_sma_200,
                    'golden_cross': golden_cross,
                    'death_cross': death_cross,
                },
                # Momentum
                'rsi': {
                    'current': rsi_val,
                    'zone': rsi_zone,
                },
                'macd': {
                    'histogram': macd_hist,
                    'signal': macd_signal,
                },
                # Volatility
                'bollinger_bands': {
                    'position': bb_position,
                    'squeeze': bb_squeeze,
                },
                'atr': {
                    'current': atr_val,
                    'atr_pct': atr_pct,
                },
                # Volume
                'volume': {
                    'current': int(volume.iloc[-1]),
                    'sma_20': vol_sma20_val,
                    'volume_ratio': volume_ratio,
                    'high_volume': high_volume,
                },
                # Levels
                'pivot_points': pivots,
                'nearest_levels': {
                    'support': nearest_support,
                    'resistance': nearest_resistance,
                },
                # Synthesis
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'overall_signal': overall_signal,
                'confidence': confidence,
            }

            return result

        except Exception as e:
            return {
                'error': f"Technical analysis failed for {ticker}: {str(e)}",
                'ticker': ticker.upper()
            }

    # ------------------------- helpers ------------------------- #

    def _pct_change(self, close: pd.Series, periods: int) -> float:
        if len(close) <= periods:
            return 0.0
        prev = float(close.iloc[-periods-1])
        curr = float(close.iloc[-1])
        if prev == 0:
            return 0.0
        return (curr - prev) / prev * 100.0

    def _crossed_above(self, fast: pd.Series, slow: pd.Series) -> bool:
        try:
            if len(fast) < 2 or len(slow) < 2:
                return False
            return fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
        except Exception:
            return False

    def _crossed_below(self, fast: pd.Series, slow: pd.Series) -> bool:
        try:
            if len(fast) < 2 or len(slow) < 2:
                return False
            return fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]
        except Exception:
            return False

    def _pivot_points(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, float]:
        high_last = float(high.iloc[-1])
        low_last = float(low.iloc[-1])
        close_last = float(close.iloc[-1])
        pivot = (high_last + low_last + close_last) / 3.0
        r1 = 2 * pivot - low_last
        s1 = 2 * pivot - high_last
        r2 = pivot + (high_last - low_last)
        s2 = pivot - (high_last - low_last)
        return {
            'pivot': pivot,
            'resistance_1': r1,
            'resistance_2': r2,
            'support_1': s1,
            'support_2': s2,
        }

    def _nearest_levels(self, high: pd.Series, low: pd.Series, close: pd.Series) -> (float, float):
        # Use recent 20-day highs/lows as dynamic levels
        recent_highs = high.tail(20)
        recent_lows = low.tail(20)
        nearest_res = float(recent_highs.min()) if not recent_highs.empty else float(high.iloc[-1])
        nearest_sup = float(recent_lows.max()) if not recent_lows.empty else float(low.iloc[-1])
        return nearest_sup, nearest_res

    def _collect_signals(
        self,
        rsi_val: float,
        rsi_zone: str,
        macd_hist: float,
        price_vs_sma_50: str,
        bb_position: str,
        volume_ratio: float,
        golden_cross: bool,
        death_cross: bool,
    ) -> (list, list):
        buy_signals = []
        sell_signals = []

        if rsi_zone == 'oversold':
            buy_signals.append('RSI oversold (<30)')
        if rsi_zone == 'overbought':
            sell_signals.append('RSI overbought (>70)')

        if macd_hist > 0:
            buy_signals.append('MACD histogram positive')
        elif macd_hist < 0:
            sell_signals.append('MACD histogram negative')

        if price_vs_sma_50 == 'above':
            buy_signals.append('Price above SMA50 trend')
        else:
            sell_signals.append('Price below SMA50 trend')

        if bb_position == 'lower':
            buy_signals.append('Price near lower Bollinger band (reversion)')
        elif bb_position == 'upper':
            sell_signals.append('Price near upper Bollinger band (reversion)')

        if volume_ratio > 1.5:
            buy_signals.append('High volume surge (ratio > 1.5)')

        if golden_cross:
            buy_signals.append('Golden cross (SMA20 > SMA50)')
        if death_cross:
            sell_signals.append('Death cross (SMA20 < SMA50)')

        return buy_signals, sell_signals

    def _baseline_verdict(self, buys: list, sells: list) -> (str, float):
        b = len(buys)
        s = len(sells)
        if b > s + 1:
            signal = 'BUY'
        elif s > b + 1:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        total = max(1, b + s)
        confidence = round(50 + (abs(b - s) / total) * 50 * 0.6, 2)  # cap confidence gain
        return signal, confidence


