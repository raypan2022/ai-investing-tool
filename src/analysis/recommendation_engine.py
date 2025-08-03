"""
Recommendation Engine for Stock Analysis

This module provides a comprehensive recommendation engine that combines
technical analysis, fundamental data, economic context, and news sentiment
to generate both short-term and long-term investment recommendations.
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Enumeration for recommendation types"""
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class ConfidenceLevel(Enum):
    """Enumeration for confidence levels"""
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"


@dataclass
class FactorScore:
    """Data class for factor scores"""
    technical: float
    fundamental: float
    economic: float
    news_sentiment: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {
            'technical': self.technical,
            'fundamental': self.fundamental,
            'economic': self.economic,
            'news_sentiment': self.news_sentiment
        }


@dataclass
class Recommendation:
    """Data class for investment recommendations"""
    recommendation: RecommendationType
    confidence: ConfidenceLevel
    score: float
    timeframe: str
    factor_scores: FactorScore
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'recommendation': self.recommendation.value,
            'confidence': self.confidence.value,
            'score': self.score,
            'timeframe': self.timeframe,
            'factor_scores': self.factor_scores.to_dict()
        }


class RecommendationEngine:
    """
    Comprehensive recommendation engine for stock analysis.
    
    This engine combines multiple data sources to generate both short-term
    and long-term investment recommendations with confidence levels.
    """
    
    # Weight configurations for different timeframes
    SHORT_TERM_WEIGHTS = {
        'technical': 0.5,      # Price patterns and timing
        'fundamental': 0.2,    # Company quality and valuation
        'economic': 0.15,      # Market environment
        'news_sentiment': 0.15 # News and filings sentiment
    }
    
    LONG_TERM_WEIGHTS = {
        'fundamental': 0.5,    # Company quality paramount
        'news_sentiment': 0.2, # Company news and filings
        'technical': 0.2,      # Entry timing
        'economic': 0.1        # Economic cycles
    }
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 20
    MODERATE_CONFIDENCE_THRESHOLD = 10
    
    def __init__(self):
        """Initialize the recommendation engine"""
        self.logger = logging.getLogger(__name__)
    
    def generate_recommendations(
        self,
        fundamental_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        economic_context: Dict[str, Any],
        rag_insights: Dict[str, Any]
    ) -> Dict[str, Recommendation]:
        """
        Generate both short-term and long-term recommendations.
        
        Args:
            fundamental_data: Company fundamental data
            technical_analysis: Technical analysis results
            economic_context: Economic indicators and context
            rag_insights: News and filings analysis
            
        Returns:
            Dictionary containing short-term and long-term recommendations
        """
        try:
            # Generate short-term recommendation
            short_term = self._generate_short_term_recommendation(
                fundamental_data, technical_analysis, economic_context, rag_insights
            )
            
            # Generate long-term recommendation
            long_term = self._generate_long_term_recommendation(
                fundamental_data, technical_analysis, economic_context, rag_insights
            )
            
            return {
                'short_term': short_term,
                'long_term': long_term
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            raise
    
    def _generate_short_term_recommendation(
        self,
        fundamental_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        economic_context: Dict[str, Any],
        rag_insights: Dict[str, Any]
    ) -> Recommendation:
        """Generate short-term recommendation (1-2 weeks)"""
        
        # Calculate individual factor scores
        technical_score = self._calculate_technical_score(technical_analysis)
        fundamental_score = self._calculate_fundamental_score(fundamental_data)
        economic_score = self._calculate_economic_score(economic_context)
        news_score = self._calculate_news_sentiment_score(rag_insights)
        
        # Create factor scores object
        factor_scores = FactorScore(
            technical=technical_score,
            fundamental=fundamental_score,
            economic=economic_score,
            news_sentiment=news_score
        )
        
        # Calculate weighted final score
        final_score = (
            technical_score * self.SHORT_TERM_WEIGHTS['technical'] +
            fundamental_score * self.SHORT_TERM_WEIGHTS['fundamental'] +
            economic_score * self.SHORT_TERM_WEIGHTS['economic'] +
            news_score * self.SHORT_TERM_WEIGHTS['news_sentiment']
        )
        
        # Determine recommendation and confidence
        recommendation = self._get_recommendation(final_score)
        confidence = self._get_confidence_level(final_score, self.HIGH_CONFIDENCE_THRESHOLD)
        
        return Recommendation(
            recommendation=recommendation,
            confidence=confidence,
            score=final_score,
            timeframe="1-2 weeks",
            factor_scores=factor_scores
        )
    
    def _generate_long_term_recommendation(
        self,
        fundamental_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        economic_context: Dict[str, Any],
        rag_insights: Dict[str, Any]
    ) -> Recommendation:
        """Generate long-term recommendation (6-12 months)"""
        
        # Calculate individual factor scores
        fundamental_score = self._calculate_fundamental_score(fundamental_data)
        news_score = self._calculate_news_sentiment_score(rag_insights)
        technical_score = self._calculate_technical_score(technical_analysis)
        economic_score = self._calculate_economic_score(economic_context)
        
        # Create factor scores object
        factor_scores = FactorScore(
            technical=technical_score,
            fundamental=fundamental_score,
            economic=economic_score,
            news_sentiment=news_score
        )
        
        # Calculate weighted final score
        final_score = (
            fundamental_score * self.LONG_TERM_WEIGHTS['fundamental'] +
            news_score * self.LONG_TERM_WEIGHTS['news_sentiment'] +
            technical_score * self.LONG_TERM_WEIGHTS['technical'] +
            economic_score * self.LONG_TERM_WEIGHTS['economic']
        )
        
        # Determine recommendation and confidence
        recommendation = self._get_recommendation(final_score)
        confidence = self._get_confidence_level(final_score, self.HIGH_CONFIDENCE_THRESHOLD)
        
        return Recommendation(
            recommendation=recommendation,
            confidence=confidence,
            score=final_score,
            timeframe="6-12 months",
            factor_scores=factor_scores
        )
    
    def _calculate_technical_score(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate technical analysis score (0-100).
        
        Args:
            analysis: Technical analysis results
            
        Returns:
            Technical score between 0 and 100
        """
        try:
            score = 50.0  # Neutral starting point
            
            # Overall signal analysis
            signal = analysis.get('overall_signal', 'HOLD')
            confidence = analysis.get('confidence', 50.0)
            
            if signal == 'BUY':
                score += (confidence - 50.0) * 0.5
            elif signal == 'SELL':
                score -= (confidence - 50.0) * 0.5
            
            # Z-score analysis (mean reversion)
            z_score_analysis = analysis.get('advanced_statistics', {}).get('z_score_analysis', {})
            price_z_score = z_score_analysis.get('price_z_score', 0.0)
            
            if abs(price_z_score) > 2.0:
                if price_z_score < -2.0:  # Oversold
                    score += 15.0
                else:  # Overbought
                    score -= 15.0
            
            # RSI analysis
            rsi_data = analysis.get('rsi', {})
            current_rsi = rsi_data.get('current', 50.0)
            
            if current_rsi < 30.0:
                score += 10.0  # Oversold
            elif current_rsi > 70.0:
                score -= 10.0  # Overbought
            
            # MACD analysis
            macd_data = analysis.get('macd', {})
            macd_signal = macd_data.get('signal', 'neutral')
            
            if macd_signal == 'bullish':
                score += 5.0
            elif macd_signal == 'bearish':
                score -= 5.0
            
            # Moving average analysis
            ma_data = analysis.get('moving_averages', {})
            price_vs_sma20 = ma_data.get('price_vs_sma20', 0.0)
            
            if price_vs_sma20 > 5.0:  # Above SMA20
                score += 5.0
            elif price_vs_sma20 < -5.0:  # Below SMA20
                score -= 5.0
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating technical score: {e}")
            return 50.0
    
    def _calculate_fundamental_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate fundamental analysis score (0-100).
        
        Args:
            data: Fundamental data
            
        Returns:
            Fundamental score between 0 and 100
        """
        try:
            score = 50.0  # Neutral starting point
            
            # PE ratio analysis
            pe_ratio = data.get('pe_ratio', 25.0)
            if 15.0 <= pe_ratio <= 25.0:
                score += 15.0  # Good value
            elif pe_ratio < 15.0:
                score += 25.0  # Undervalued
            elif pe_ratio > 30.0:
                score -= 15.0  # Overvalued
            
            # Price momentum analysis
            price_changes = data.get('price_changes', {})
            momentum_20d = price_changes.get('20d', 0.0)
            
            if momentum_20d > 5.0:
                score += 10.0  # Positive momentum
            elif momentum_20d < -5.0:
                score -= 10.0  # Negative momentum
            
            # Market cap analysis (stability factor)
            market_cap = data.get('market_cap', 0.0)
            if market_cap > 100_000_000_000:  # > $100B
                score += 5.0  # Large cap stability
            elif market_cap < 10_000_000_000:  # < $10B
                score -= 5.0  # Small cap volatility
            
            # Volume analysis
            avg_volume = data.get('avg_volume', 0.0)
            if avg_volume > 10_000_000:  # High liquidity
                score += 5.0
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating fundamental score: {e}")
            return 50.0
    
    def _calculate_economic_score(self, context: Dict[str, Any]) -> float:
        """
        Calculate economic context score (0-100).
        
        Args:
            context: Economic indicators and context
            
        Returns:
            Economic score between 0 and 100
        """
        try:
            score = 50.0  # Neutral starting point
            
            # Economic sentiment analysis
            economic_indicators = context.get('economic_indicators', {})
            sentiment_data = economic_indicators.get('economic_sentiment', {})
            overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
            
            sentiment_scores = {
                'very_bullish': 20.0,
                'bullish': 10.0,
                'neutral': 0.0,
                'bearish': -10.0,
                'very_bearish': -20.0
            }
            score += sentiment_scores.get(overall_sentiment, 0.0)
            
            # Fed policy analysis
            fed_rate = economic_indicators.get('fed_rate', {})
            change_direction = fed_rate.get('change_direction', 'neutral')
            
            if change_direction == 'cut':
                score += 10.0  # Dovish policy
            elif change_direction == 'hike':
                score -= 10.0  # Hawkish policy
            
            # Market volatility analysis
            volatility = economic_indicators.get('market_volatility', {})
            volatility_regime = volatility.get('volatility_regime', 'normal')
            
            if volatility_regime == 'low':
                score += 5.0  # Favorable low volatility
            elif volatility_regime == 'high':
                score -= 5.0  # Unfavorable high volatility
            
            # Inflation analysis
            inflation = economic_indicators.get('inflation', {})
            inflation_trend = inflation.get('trend', 'stable')
            
            if inflation_trend == 'decreasing':
                score += 5.0  # Favorable
            elif inflation_trend == 'increasing':
                score -= 5.0  # Unfavorable
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating economic score: {e}")
            return 50.0
    
    def _calculate_news_sentiment_score(self, rag_insights: Dict[str, Any]) -> float:
        """
        Calculate news sentiment score (0-100).
        
        Args:
            rag_insights: News and filings analysis
            
        Returns:
            News sentiment score between 0 and 100
        """
        try:
            score = 50.0  # Neutral starting point
            
            # Market sentiment analysis
            market_sentiment = rag_insights.get('market_sentiment', 'neutral')
            sentiment_scores = {
                'positive': 15.0,
                'neutral': 0.0,
                'negative': -15.0
            }
            score += sentiment_scores.get(market_sentiment, 0.0)
            
            # Key themes analysis
            key_themes = rag_insights.get('key_themes', [])
            positive_themes = ['growth', 'expansion', 'innovation', 'strong', 'increase', 'profit', 'revenue']
            negative_themes = ['decline', 'weak', 'decrease', 'risk', 'uncertainty', 'loss', 'debt']
            
            theme_score = 0.0
            for theme in key_themes:
                theme_lower = theme.lower()
                if any(pos in theme_lower for pos in positive_themes):
                    theme_score += 5.0
                elif any(neg in theme_lower for neg in negative_themes):
                    theme_score -= 5.0
            
            score += min(20.0, max(-20.0, theme_score))  # Cap theme impact
            
            # Relevant documents analysis
            relevant_docs = rag_insights.get('relevant_documents', [])
            high_relevance_count = sum(
                1 for doc in relevant_docs 
                if doc.get('relevance_score', 0.0) > 0.8
            )
            
            if high_relevance_count >= 2:
                score += 10.0  # Strong relevant news
            elif high_relevance_count == 0:
                score -= 5.0   # No relevant news
            
            # Fundamental summary analysis
            fundamental_summary = rag_insights.get('fundamental_summary', '')
            if fundamental_summary:
                if any(pos in fundamental_summary.lower() for pos in positive_themes):
                    score += 5.0
                elif any(neg in fundamental_summary.lower() for neg in negative_themes):
                    score -= 5.0
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating news sentiment score: {e}")
            return 50.0
    
    def _get_recommendation(self, score: float) -> RecommendationType:
        """
        Convert score to recommendation type.
        
        Args:
            score: Final weighted score (0-100)
            
        Returns:
            Recommendation type
        """
        if score >= 70.0:
            return RecommendationType.BUY
        elif score <= 30.0:
            return RecommendationType.SELL
        else:
            return RecommendationType.HOLD
    
    def _get_confidence_level(self, score: float, high_threshold: float) -> ConfidenceLevel:
        """
        Determine confidence level based on score deviation from neutral.
        
        Args:
            score: Final weighted score (0-100)
            high_threshold: Threshold for high confidence
            
        Returns:
            Confidence level
        """
        deviation = abs(score - 50.0)
        
        if deviation >= high_threshold:
            return ConfidenceLevel.HIGH
        elif deviation >= self.MODERATE_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MODERATE
        else:
            return ConfidenceLevel.LOW
    
    def get_recommendation_summary(self, recommendations: Dict[str, Recommendation]) -> Dict[str, Any]:
        """
        Generate a summary of recommendations for LLM input.
        
        Args:
            recommendations: Dictionary containing short-term and long-term recommendations
            
        Returns:
            Summary dictionary for LLM processing
        """
        return {
            'short_term': recommendations['short_term'].to_dict(),
            'long_term': recommendations['long_term'].to_dict(),
            'summary': {
                'overall_sentiment': self._get_overall_sentiment(recommendations),
                'confidence_agreement': self._get_confidence_agreement(recommendations),
                'factor_consistency': self._get_factor_consistency(recommendations)
            }
        }
    
    def _get_overall_sentiment(self, recommendations: Dict[str, Recommendation]) -> str:
        """Determine overall sentiment across timeframes"""
        short_score = recommendations['short_term'].score
        long_score = recommendations['long_term'].score
        
        avg_score = (short_score + long_score) / 2
        
        if avg_score >= 70:
            return 'bullish'
        elif avg_score <= 30:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_confidence_agreement(self, recommendations: Dict[str, Recommendation]) -> str:
        """Check if confidence levels agree across timeframes"""
        short_conf = recommendations['short_term'].confidence
        long_conf = recommendations['long_term'].confidence
        
        if short_conf == long_conf:
            return 'agreement'
        elif (short_conf == ConfidenceLevel.HIGH and long_conf == ConfidenceLevel.MODERATE) or \
             (short_conf == ConfidenceLevel.MODERATE and long_conf == ConfidenceLevel.HIGH):
            return 'partial_agreement'
        else:
            return 'disagreement'
    
    def _get_factor_consistency(self, recommendations: Dict[str, Recommendation]) -> Dict[str, float]:
        """Calculate consistency between factor scores across timeframes"""
        short_factors = recommendations['short_term'].factor_scores
        long_factors = recommendations['long_term'].factor_scores
        
        return {
            'technical': abs(short_factors.technical - long_factors.technical),
            'fundamental': abs(short_factors.fundamental - long_factors.fundamental),
            'economic': abs(short_factors.economic - long_factors.economic),
            'news_sentiment': abs(short_factors.news_sentiment - long_factors.news_sentiment)
        } 