import requests
import pandas as pd
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import yfinance as yf

class EconomicNewsFetcher:
    """Fetches economic indicators, monetary policy data, and political events"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.stlouisfed.org/fred/series"
        
    def get_economic_indicators(self) -> Dict[str, Any]:
        """Get key economic indicators"""
        try:
            indicators = {}
            
            # Federal Funds Rate (Fed Rate)
            indicators['fed_rate'] = self._get_fed_rate()
            
            # Inflation (CPI)
            indicators['inflation'] = self._get_inflation_data()
            
            # Unemployment Rate
            indicators['unemployment'] = self._get_unemployment_data()
            
            # GDP Growth
            indicators['gdp_growth'] = self._get_gdp_data()
            
            # Market Volatility (VIX)
            indicators['market_volatility'] = self._get_vix_data()
            
            # Treasury Yields
            indicators['treasury_yields'] = self._get_treasury_yields()
            
            # Economic Sentiment
            indicators['economic_sentiment'] = self._analyze_economic_sentiment(indicators)
            
            return indicators
            
        except Exception as e:
            print(f"Error fetching economic indicators: {e}")
            return self._get_mock_economic_data()
    
    def _get_fed_rate(self) -> Dict[str, Any]:
        """Get current Federal Funds Rate and expectations"""
        try:
            # For MVP, use mock data
            # In production, fetch from FRED API
            return {
                'current_rate': 5.25,
                'last_change': '2024-07-31',
                'change_direction': 'hold',
                'next_meeting': '2024-09-18',
                'rate_expectations': {
                    'next_meeting': 'hold',
                    'year_end': 5.0,
                    'next_year': 4.5
                },
                'market_probability': {
                    'hold': 0.85,
                    'cut_25bp': 0.15,
                    'hike_25bp': 0.0
                }
            }
        except Exception as e:
            print(f"Error fetching Fed rate: {e}")
            return {'current_rate': 5.25, 'change_direction': 'hold'}
    
    def _get_inflation_data(self) -> Dict[str, Any]:
        """Get inflation metrics"""
        try:
            return {
                'cpi_yoy': 3.2,
                'core_cpi_yoy': 3.8,
                'pce_yoy': 2.6,
                'core_pce_yoy': 2.8,
                'trend': 'decreasing',
                'fed_target': 2.0,
                'last_update': '2024-07-15'
            }
        except Exception as e:
            print(f"Error fetching inflation data: {e}")
            return {'cpi_yoy': 3.2, 'trend': 'decreasing'}
    
    def _get_unemployment_data(self) -> Dict[str, Any]:
        """Get unemployment metrics"""
        try:
            return {
                'unemployment_rate': 3.9,
                'labor_force_participation': 62.6,
                'job_growth': 187000,
                'wage_growth_yoy': 4.1,
                'trend': 'stable',
                'last_update': '2024-08-02'
            }
        except Exception as e:
            print(f"Error fetching unemployment data: {e}")
            return {'unemployment_rate': 3.9, 'trend': 'stable'}
    
    def _get_gdp_data(self) -> Dict[str, Any]:
        """Get GDP growth data"""
        try:
            return {
                'q2_2024': 2.1,
                'q1_2024': 1.4,
                'q4_2023': 3.4,
                'annual_2023': 2.5,
                'forecast_2024': 2.2,
                'trend': 'moderate_growth',
                'last_update': '2024-07-25'
            }
        except Exception as e:
            print(f"Error fetching GDP data: {e}")
            return {'q2_2024': 2.1, 'trend': 'moderate_growth'}
    
    def _get_vix_data(self) -> Dict[str, Any]:
        """Get market volatility data"""
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1mo")
            current_vix = vix_data['Close'].iloc[-1]
            
            return {
                'current_vix': current_vix,
                'vix_20d_avg': vix_data['Close'].rolling(20).mean().iloc[-1],
                'volatility_regime': 'low' if current_vix < 15 else 'moderate' if current_vix < 25 else 'high',
                'fear_greed_index': self._calculate_fear_greed(current_vix),
                'last_update': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return {'current_vix': 15.5, 'volatility_regime': 'low'}
    
    def _get_treasury_yields(self) -> Dict[str, Any]:
        """Get Treasury yield curve data"""
        try:
            # Get yield curve data
            yields = {}
            for maturity in ['^TNX', '^TYX', '^IRX']:
                ticker = yf.Ticker(maturity)
                data = ticker.history(period="1d")
                if not data.empty:
                    if maturity == '^TNX':  # 10-year
                        yields['10y'] = data['Close'].iloc[-1]
                    elif maturity == '^TYX':  # 30-year
                        yields['30y'] = data['Close'].iloc[-1]
                    elif maturity == '^IRX':  # 13-week
                        yields['3m'] = data['Close'].iloc[-1]
            
            # Calculate yield curve inversion
            curve_inverted = yields.get('10y', 0) < yields.get('3m', 0)
            
            return {
                '3m_yield': yields.get('3m', 5.4),
                '10y_yield': yields.get('10y', 4.2),
                '30y_yield': yields.get('30y', 4.4),
                'curve_inverted': curve_inverted,
                'spread_10y_3m': yields.get('10y', 4.2) - yields.get('3m', 5.4),
                'last_update': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            print(f"Error fetching Treasury yields: {e}")
            return {
                '3m_yield': 5.4,
                '10y_yield': 4.2,
                'curve_inverted': True,
                'spread_10y_3m': -1.2
            }
    
    def _calculate_fear_greed(self, vix: float) -> str:
        """Calculate fear/greed index based on VIX"""
        if vix < 12:
            return "extreme_greed"
        elif vix < 15:
            return "greed"
        elif vix < 20:
            return "neutral"
        elif vix < 25:
            return "fear"
        else:
            return "extreme_fear"
    
    def _analyze_economic_sentiment(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall economic sentiment"""
        try:
            # Analyze various indicators
            fed_rate = indicators.get('fed_rate', {})
            inflation = indicators.get('inflation', {})
            unemployment = indicators.get('unemployment', {})
            gdp = indicators.get('gdp_data', {})
            vix = indicators.get('market_volatility', {})
            yields = indicators.get('treasury_yields', {})
            
            # Calculate sentiment score
            sentiment_score = 0
            factors = []
            
            # Fed policy (positive if dovish, negative if hawkish)
            if fed_rate.get('change_direction') == 'cut':
                sentiment_score += 20
                factors.append('dovish_fed')
            elif fed_rate.get('change_direction') == 'hike':
                sentiment_score -= 20
                factors.append('hawkish_fed')
            
            # Inflation (positive if decreasing)
            if inflation.get('trend') == 'decreasing':
                sentiment_score += 15
                factors.append('decreasing_inflation')
            elif inflation.get('trend') == 'increasing':
                sentiment_score -= 15
                factors.append('increasing_inflation')
            
            # Unemployment (positive if low and stable)
            if unemployment.get('unemployment_rate', 10) < 4.0:
                sentiment_score += 10
                factors.append('low_unemployment')
            
            # GDP growth (positive if growing)
            if gdp.get('q2_2024', 0) > 2.0:
                sentiment_score += 10
                factors.append('strong_gdp')
            
            # Market volatility (positive if low)
            if vix.get('volatility_regime') == 'low':
                sentiment_score += 10
                factors.append('low_volatility')
            
            # Yield curve (negative if inverted)
            if yields.get('curve_inverted', False):
                sentiment_score -= 15
                factors.append('inverted_yield_curve')
            
            # Determine overall sentiment
            if sentiment_score >= 30:
                sentiment = 'very_bullish'
            elif sentiment_score >= 15:
                sentiment = 'bullish'
            elif sentiment_score >= -15:
                sentiment = 'neutral'
            elif sentiment_score >= -30:
                sentiment = 'bearish'
            else:
                sentiment = 'very_bearish'
            
            return {
                'sentiment_score': sentiment_score,
                'overall_sentiment': sentiment,
                'key_factors': factors,
                'market_implications': self._get_market_implications(sentiment, factors)
            }
            
        except Exception as e:
            print(f"Error analyzing economic sentiment: {e}")
            return {
                'sentiment_score': 0,
                'overall_sentiment': 'neutral',
                'key_factors': ['data_unavailable'],
                'market_implications': 'insufficient_data'
            }
    
    def _get_market_implications(self, sentiment: str, factors: List[str]) -> Dict[str, Any]:
        """Get market implications of economic sentiment"""
        implications = {
            'very_bullish': {
                'equity_outlook': 'strong_buy',
                'bond_outlook': 'sell',
                'sector_favorites': ['technology', 'consumer_discretionary', 'financials'],
                'risk_level': 'low',
                'positioning': 'aggressive'
            },
            'bullish': {
                'equity_outlook': 'buy',
                'bond_outlook': 'underweight',
                'sector_favorites': ['technology', 'healthcare', 'consumer_discretionary'],
                'risk_level': 'low_to_moderate',
                'positioning': 'moderate_to_aggressive'
            },
            'neutral': {
                'equity_outlook': 'hold',
                'bond_outlook': 'neutral',
                'sector_favorites': ['utilities', 'consumer_staples', 'healthcare'],
                'risk_level': 'moderate',
                'positioning': 'balanced'
            },
            'bearish': {
                'equity_outlook': 'sell',
                'bond_outlook': 'buy',
                'sector_favorites': ['utilities', 'consumer_staples', 'defensive'],
                'risk_level': 'high',
                'positioning': 'defensive'
            },
            'very_bearish': {
                'equity_outlook': 'strong_sell',
                'bond_outlook': 'strong_buy',
                'sector_favorites': ['utilities', 'consumer_staples', 'cash'],
                'risk_level': 'very_high',
                'positioning': 'very_defensive'
            }
        }
        
        return implications.get(sentiment, implications['neutral'])
    
    def get_political_events(self) -> Dict[str, Any]:
        """Get political events and their market impact"""
        try:
            # For MVP, use mock data
            # In production, fetch from news APIs
            return {
                'election_events': {
                    'presidential_election': '2024-11-05',
                    'market_impact': 'moderate',
                    'key_issues': ['tax_policy', 'regulation', 'trade_policy'],
                    'uncertainty_level': 'high'
                },
                'policy_events': {
                    'fiscal_policy': 'stable',
                    'regulatory_changes': 'moderate',
                    'trade_policy': 'stable',
                    'key_legislation': ['infrastructure_bill', 'climate_policy']
                },
                'geopolitical_events': {
                    'trade_tensions': 'moderate',
                    'international_conflicts': 'low',
                    'supply_chain_risks': 'moderate'
                },
                'market_impact': {
                    'overall_impact': 'neutral',
                    'sector_impacts': {
                        'technology': 'positive',
                        'energy': 'negative',
                        'healthcare': 'neutral',
                        'financials': 'neutral'
                    }
                }
            }
        except Exception as e:
            print(f"Error fetching political events: {e}")
            return {'market_impact': {'overall_impact': 'neutral'}}
    
    def _get_mock_economic_data(self) -> Dict[str, Any]:
        """Return mock economic data for development"""
        return {
            'fed_rate': {
                'current_rate': 5.25,
                'change_direction': 'hold',
                'next_meeting': '2024-09-18'
            },
            'inflation': {
                'cpi_yoy': 3.2,
                'trend': 'decreasing'
            },
            'unemployment': {
                'unemployment_rate': 3.9,
                'trend': 'stable'
            },
            'gdp_growth': {
                'q2_2024': 2.1,
                'trend': 'moderate_growth'
            },
            'market_volatility': {
                'current_vix': 15.5,
                'volatility_regime': 'low'
            },
            'treasury_yields': {
                '10y_yield': 4.2,
                'curve_inverted': True
            },
            'economic_sentiment': {
                'sentiment_score': 10,
                'overall_sentiment': 'bullish',
                'key_factors': ['decreasing_inflation', 'low_unemployment']
            }
        } 