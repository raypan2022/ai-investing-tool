"""
LLM Analyzer for Professional Investment Recommendations

This module generates professional, actionable investment recommendations
in a structured format suitable for institutional use.
"""

from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    Generates professional investment recommendations in structured format.
    
    This class takes the recommendation engine output and generates
    professional, actionable investment advice with specific entry points,
    targets, and risk management.
    """
    
    def __init__(self):
        """Initialize the LLM analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def generate_recommendation_report(
        self,
        ticker: str,
        recommendations: Dict[str, Any],
        data: Dict[str, Any]
    ) -> str:
        """
        Generate a complete professional investment recommendation report.
        
        Args:
            ticker: Stock ticker symbol
            recommendations: Output from recommendation engine
            data: Complete analysis data including technical, economic, and news
            
        Returns:
            Formatted recommendation report
        """
        try:
            # Extract key data
            current_price = data['current_price']
            short_term = recommendations['short_term']
            long_term = recommendations['long_term']
            
            # Generate the report
            report = self._format_report_header(ticker, current_price)
            report += self._format_short_term_outlook(short_term, data)
            report += self._format_long_term_outlook(long_term, data)
            report += self._format_risk_section(data)
            report += self._format_monitoring_section(data)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation report: {e}")
            return f"Error generating recommendation report: {e}"
    
    def _format_report_header(self, ticker: str, current_price: float) -> str:
        """Format the report header"""
        return f"""INVESTMENT RECOMMENDATION REPORT
Ticker: {ticker.upper()}
Current Price: ${current_price:.2f}
Date: {datetime.now().strftime('%B %d, %Y')}

"""
    
    def _format_short_term_outlook(self, short_term: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Format the short-term outlook section"""
        recommendation = short_term['recommendation']
        confidence = short_term['confidence']
        score = short_term['score']
        
        # Generate summary based on factors
        summary = self._generate_short_term_summary(short_term, data)
        
        # Generate evidence points
        evidence = self._generate_short_term_evidence(short_term, data)
        
        # Generate action plan
        action_plan = self._generate_short_term_action_plan(short_term, data)
        
        return f"""SHORT-TERM OUTLOOK (1-2 weeks):
RECOMMENDATION: {recommendation}
CONFIDENCE: {confidence}

SUMMARY: {summary}

EVIDENCE:
{evidence}

ACTION PLAN:
{action_plan}

"""
    
    def _format_long_term_outlook(self, long_term: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Format the long-term outlook section"""
        recommendation = long_term['recommendation']
        confidence = long_term['confidence']
        score = long_term['score']
        
        # Generate summary based on factors
        summary = self._generate_long_term_summary(long_term, data)
        
        # Generate evidence points
        evidence = self._generate_long_term_evidence(long_term, data)
        
        # Generate action plan
        action_plan = self._generate_long_term_action_plan(long_term, data)
        
        return f"""LONG-TERM OUTLOOK (6-12 months):
RECOMMENDATION: {recommendation}
CONFIDENCE: {confidence}

SUMMARY: {summary}

EVIDENCE:
{evidence}

ACTION PLAN:
{action_plan}

"""
    
    def _generate_short_term_summary(self, short_term: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Generate short-term summary based on factor scores"""
        technical_score = short_term['factor_scores']['technical']
        fundamental_score = short_term['factor_scores']['fundamental']
        economic_score = short_term['factor_scores']['economic']
        news_score = short_term['factor_scores']['news_sentiment']
        
        # Determine primary drivers
        drivers = []
        if technical_score > 70:
            drivers.append("oversold technical conditions")
        elif technical_score < 30:
            drivers.append("overbought technical conditions")
        
        if news_score > 70:
            drivers.append("positive news sentiment")
        elif news_score < 30:
            drivers.append("negative news sentiment")
        
        if economic_score > 70:
            drivers.append("favorable economic backdrop")
        elif economic_score < 30:
            drivers.append("challenging economic environment")
        
        if fundamental_score > 70:
            drivers.append("strong fundamentals")
        elif fundamental_score < 30:
            drivers.append("weak fundamentals")
        
        if not drivers:
            drivers = ["mixed market signals"]
        
        recommendation = short_term['recommendation']
        
        if recommendation == 'BUY':
            if len(drivers) == 1:
                return f"Strong short-term buying opportunity driven by {drivers[0]}."
            else:
                return f"Strong short-term buying opportunity driven by {', '.join(drivers[:-1])}, and {drivers[-1]}."
        elif recommendation == 'SELL':
            if len(drivers) == 1:
                return f"Short-term selling opportunity driven by {drivers[0]}."
            else:
                return f"Short-term selling opportunity driven by {', '.join(drivers[:-1])}, and {drivers[-1]}."
        else:
            return f"Neutral short-term outlook with {', '.join(drivers)} suggesting a wait-and-see approach."
    
    def _generate_long_term_summary(self, long_term: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Generate long-term summary based on factor scores"""
        fundamental_score = long_term['factor_scores']['fundamental']
        news_score = long_term['factor_scores']['news_sentiment']
        technical_score = long_term['factor_scores']['technical']
        economic_score = long_term['factor_scores']['economic']
        
        # Determine primary drivers
        drivers = []
        if fundamental_score > 70:
            drivers.append("strong fundamentals")
        elif fundamental_score < 30:
            drivers.append("weak fundamentals")
        
        if news_score > 70:
            drivers.append("positive company news")
        elif news_score < 30:
            drivers.append("negative company news")
        
        if economic_score > 70:
            drivers.append("favorable economic environment")
        elif economic_score < 30:
            drivers.append("challenging economic environment")
        
        if technical_score > 70:
            drivers.append("attractive entry timing")
        elif technical_score < 30:
            drivers.append("poor entry timing")
        
        if not drivers:
            drivers = ["mixed long-term signals"]
        
        recommendation = long_term['recommendation']
        
        if recommendation == 'BUY':
            if len(drivers) == 1:
                return f"Excellent long-term investment opportunity supported by {drivers[0]}."
            else:
                return f"Excellent long-term investment opportunity supported by {', '.join(drivers[:-1])}, and {drivers[-1]}."
        elif recommendation == 'SELL':
            if len(drivers) == 1:
                return f"Long-term selling opportunity due to {drivers[0]}."
            else:
                return f"Long-term selling opportunity due to {', '.join(drivers[:-1])}, and {drivers[-1]}."
        else:
            return f"Neutral long-term outlook with {', '.join(drivers)} suggesting careful monitoring."
    
    def _generate_short_term_evidence(self, short_term: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Generate evidence points for short-term outlook"""
        evidence_points = []
        
        # Technical evidence
        technical_score = short_term['factor_scores']['technical']
        technical_data = data.get('technical_indicators', {})
        
        if technical_score > 70:
            rsi = technical_data.get('rsi', '50.0')
            z_score = technical_data.get('z_score', '0.0')
            evidence_points.append(f"• Technical oversold: Z-score of {z_score} indicates extreme statistical deviation")
        elif technical_score < 30:
            rsi = technical_data.get('rsi', '50.0')
            evidence_points.append(f"• Technical overbought: RSI of {rsi} suggests overvaluation")
        else:
            rsi = technical_data.get('rsi', '50.0')
            evidence_points.append(f"• Technical neutral: RSI of {rsi} indicates balanced conditions")
        
        # News sentiment evidence
        news_score = short_term['factor_scores']['news_sentiment']
        rag_insights = data.get('rag_insights', {})
        
        if news_score > 70:
            themes = rag_insights.get('key_themes', [])
            if themes:
                theme_summary = ', '.join(themes[:3])
                evidence_points.append(f"• Positive news sentiment: Strong market sentiment with key themes of {theme_summary}")
        elif news_score < 30:
            evidence_points.append("• Negative news sentiment: Market concerns reflected in recent news flow")
        else:
            evidence_points.append("• Neutral news sentiment: Mixed news flow with no clear directional bias")
        
        # Economic evidence
        economic_score = short_term['factor_scores']['economic']
        economic_context = data.get('economic_context', {})
        
        if economic_score > 70:
            sentiment = economic_context.get('sentiment', 'neutral')
            inflation = economic_context.get('inflation_trend', 'stable')
            evidence_points.append(f"• Favorable economics: {sentiment.title()} economic sentiment with {inflation} inflation")
        elif economic_score < 30:
            sentiment = economic_context.get('sentiment', 'neutral')
            evidence_points.append(f"• Challenging economics: {sentiment.title()} economic sentiment creating headwinds")
        else:
            evidence_points.append("• Neutral economics: Stable economic environment with no significant catalysts")
        
        # Fundamental evidence
        fundamental_score = short_term['factor_scores']['fundamental']
        pe_ratio = data.get('pe_ratio', 25.0)
        
        if fundamental_score > 70:
            evidence_points.append(f"• Strong fundamentals: Quality company with attractive PE ratio of {pe_ratio:.1f}")
        elif fundamental_score < 30:
            evidence_points.append(f"• Weak fundamentals: Concerning valuation with PE ratio of {pe_ratio:.1f}")
        else:
            evidence_points.append(f"• Solid fundamentals: Quality company with reasonable PE ratio of {pe_ratio:.1f}")
        
        return '\n'.join(evidence_points)
    
    def _generate_long_term_evidence(self, long_term: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Generate evidence points for long-term outlook"""
        evidence_points = []
        
        # Fundamental evidence
        fundamental_score = long_term['factor_scores']['fundamental']
        pe_ratio = data.get('pe_ratio', 25.0)
        market_cap = data.get('market_cap', 0)
        
        if fundamental_score > 70:
            evidence_points.append(f"• Strong fundamentals: Quality company with reasonable valuation and market leadership")
        elif fundamental_score < 30:
            evidence_points.append(f"• Weak fundamentals: Concerning long-term prospects with PE ratio of {pe_ratio:.1f}")
        else:
            evidence_points.append(f"• Solid fundamentals: Quality company with reasonable valuation and market leadership")
        
        # News sentiment evidence
        news_score = long_term['factor_scores']['news_sentiment']
        rag_insights = data.get('rag_insights', {})
        
        if news_score > 70:
            themes = rag_insights.get('key_themes', [])
            if themes:
                theme_summary = ', '.join(themes[:3])
                evidence_points.append(f"• Positive news sentiment: Strong market sentiment with themes of {theme_summary}")
        elif news_score < 30:
            evidence_points.append("• Negative news sentiment: Long-term concerns reflected in company news")
        else:
            evidence_points.append("• Neutral news sentiment: Mixed long-term outlook from company news")
        
        # Technical evidence
        technical_score = long_term['factor_scores']['technical']
        technical_data = data.get('technical_indicators', {})
        
        if technical_score > 70:
            evidence_points.append("• Good entry timing: Oversold technical condition provides attractive entry point")
        elif technical_score < 30:
            evidence_points.append("• Poor entry timing: Overbought technical condition suggests waiting for better entry")
        else:
            evidence_points.append("• Neutral entry timing: Technical conditions suggest standard entry approach")
        
        # Economic evidence
        economic_score = long_term['factor_scores']['economic']
        economic_context = data.get('economic_context', {})
        
        if economic_score > 70:
            sentiment = economic_context.get('sentiment', 'neutral')
            evidence_points.append(f"• Favorable economics: {sentiment.title()} economic sentiment supports growth stocks")
        elif economic_score < 30:
            sentiment = economic_context.get('sentiment', 'neutral')
            evidence_points.append(f"• Challenging economics: {sentiment.title()} economic sentiment may limit growth")
        else:
            evidence_points.append("• Stable economics: Neutral economic environment supports steady growth")
        
        return '\n'.join(evidence_points)
    
    def _generate_short_term_action_plan(self, short_term: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Generate short-term action plan"""
        current_price = data['current_price']
        recommendation = short_term['recommendation']
        confidence = short_term['confidence']
        
        if recommendation == 'BUY':
            # Calculate target and stop loss based on confidence
            if confidence == 'High':
                target_pct = 3.8
                stop_pct = 2.1
            elif confidence == 'Moderate':
                target_pct = 2.5
                stop_pct = 1.8
            else:
                target_pct = 1.5
                stop_pct = 1.2
            
            target_price = current_price * (1 + target_pct / 100)
            stop_price = current_price * (1 - stop_pct / 100)
            
            return f"""• Entry: ${current_price:.2f} (current price)
• Target: ${target_price:.2f} ({target_pct:.1f}% gain)
• Stop Loss: ${stop_price:.2f} ({stop_pct:.1f}% loss)"""
        
        elif recommendation == 'SELL':
            # Calculate target and stop loss for selling
            if confidence == 'High':
                target_pct = 3.0
                stop_pct = 2.5
            elif confidence == 'Moderate':
                target_pct = 2.0
                stop_pct = 2.0
            else:
                target_pct = 1.0
                stop_pct = 1.5
            
            target_price = current_price * (1 - target_pct / 100)
            stop_price = current_price * (1 + stop_pct / 100)
            
            return f"""• Exit: ${current_price:.2f} (current price)
• Target: ${target_price:.2f} ({target_pct:.1f}% decline)
• Stop Loss: ${stop_price:.2f} ({stop_pct:.1f}% rise)"""
        
        else:  # HOLD
            return f"""• Current Position: ${current_price:.2f} (current price)
• Strategy: Maintain current position
• Monitor: Key support at ${current_price * 0.98:.2f} and resistance at ${current_price * 1.02:.2f}"""
    
    def _generate_long_term_action_plan(self, long_term: Dict[str, Any], data: Dict[str, Any]) -> str:
        """Generate long-term action plan"""
        current_price = data['current_price']
        recommendation = long_term['recommendation']
        confidence = long_term['confidence']
        
        if recommendation == 'BUY':
            # Calculate long-term target based on confidence
            if confidence == 'High':
                target_pct = 18.6
                hold_period = 12
            elif confidence == 'Moderate':
                target_pct = 12.0
                hold_period = 9
            else:
                target_pct = 8.0
                hold_period = 6
            
            target_price = current_price * (1 + target_pct / 100)
            
            return f"""• Entry Strategy: Aggressive entry at current levels, with additional purchases on any further weakness
• Target: ${target_price:.2f} ({target_pct:.1f}% gain)
• Hold Period: {hold_period} months"""
        
        elif recommendation == 'SELL':
            # Calculate long-term target for selling
            if confidence == 'High':
                target_pct = 15.0
                hold_period = 6
            elif confidence == 'Moderate':
                target_pct = 10.0
                hold_period = 4
            else:
                target_pct = 5.0
                hold_period = 3
            
            target_price = current_price * (1 - target_pct / 100)
            
            return f"""• Exit Strategy: Gradual position reduction over {hold_period} months
• Target: ${target_price:.2f} ({target_pct:.1f}% decline)
• Timeline: {hold_period} months"""
        
        else:  # HOLD
            return f"""• Strategy: Maintain current position with periodic review
• Target: ${current_price * 1.08:.2f} (8.0% gain over 12 months)
• Hold Period: 12 months with quarterly reviews"""
    
    def _format_risk_section(self, data: Dict[str, Any]) -> str:
        """Format the risk section"""
        # Identify key risks based on data
        risks = []
        
        # Economic risks
        economic_context = data.get('economic_context', {})
        fed_policy = economic_context.get('fed_policy', 'neutral')
        if fed_policy in ['hike', 'uncertain']:
            risks.append("• Fed policy changes: Unexpected rate hikes could pressure valuations")
        
        # Technical risks
        technical_data = data.get('technical_indicators', {})
        z_score = float(technical_data.get('z_score', '0.0'))
        if abs(z_score) > 2.5:
            risks.append("• Technical extremes: Current statistical deviation may lead to mean reversion")
        
        # Fundamental risks
        pe_ratio = data.get('pe_ratio', 25.0)
        if pe_ratio > 35:
            risks.append("• Valuation risk: High PE ratio suggests limited upside potential")
        
        # News risks
        rag_insights = data.get('rag_insights', {})
        themes = rag_insights.get('key_themes', [])
        if any('competition' in theme.lower() for theme in themes):
            risks.append("• Competition: Increasing competition in key product categories")
        
        # Default risks if none identified
        if not risks:
            risks = [
                "• Market volatility: General market fluctuations may impact performance",
                "• Economic uncertainty: Macroeconomic factors could affect growth prospects"
            ]
        
        return f"""KEY RISKS:
{chr(10).join(risks)}

"""
    
    def _format_monitoring_section(self, data: Dict[str, Any]) -> str:
        """Format the monitoring section"""
        current_price = data['current_price']
        
        # Short-term monitoring
        short_resistance = current_price * 1.038  # 3.8% target
        short_support = current_price * 0.979     # 2.1% stop loss
        
        # Long-term monitoring
        long_target = current_price * 1.186       # 18.6% target
        quarterly_review = "quarterly earnings, AI developments, and emerging market expansion"
        
        return f"""MONITORING:
• Short-term: Watch for technical breakout above ${short_resistance:.2f} resistance
• Long-term: Monitor {quarterly_review}

""" 