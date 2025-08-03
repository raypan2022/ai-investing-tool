#!/usr/bin/env python3
"""
Test script for Professional Investment Recommendation Format

This script demonstrates the new professional format that generates
actionable investment recommendations in institutional-quality format.
"""

import sys
import os

from src.analysis.recommendation_engine import RecommendationEngine
from src.analysis.llm_analyzer import LLMAnalyzer


def create_sample_data():
    """Create sample data for testing the professional format"""
    
    # Sample fundamental data
    fundamental_data = {
        'current_price': 202.38,
        'market_cap': 3003400323072,
        'pe_ratio': 30.66,
        'price_changes': {
            '20d': 2.5,
            '60d': -1.2,
            '1y': 15.8
        },
        'avg_volume': 50000000
    }
    
    # Sample technical analysis
    technical_analysis = {
        'overall_signal': 'BUY',
        'confidence': 75.0,
        'rsi': {
            'current': 35.5,
            'signal': 'oversold'
        },
        'macd': {
            'signal': 'bullish',
            'histogram': 0.5
        },
        'moving_averages': {
            'price_vs_sma20': -2.5,
            'price_vs_sma50': 1.2
        },
        'advanced_statistics': {
            'z_score_analysis': {
                'price_z_score': -2.98
            }
        }
    }
    
    # Sample economic context
    economic_context = {
        'economic_indicators': {
            'economic_sentiment': {
                'overall_sentiment': 'very_bullish'
            },
            'fed_rate': {
                'change_direction': 'cut'
            },
            'market_volatility': {
                'volatility_regime': 'low'
            },
            'inflation': {
                'trend': 'decreasing'
            }
        }
    }
    
    # Sample RAG insights
    rag_insights = {
        'market_sentiment': 'positive',
        'key_themes': [
            'AI investment and development',
            'Emerging markets expansion',
            'Supply chain optimization',
            'Strong revenue growth',
            'Innovation in product development'
        ],
        'relevant_documents': [
            {
                'title': 'Q4 Earnings Report',
                'content': 'Strong quarterly results with revenue growth of 15%...',
                'relevance_score': 0.92,
                'source': 'SEC Filing'
            },
            {
                'title': 'AI Strategy Update',
                'content': 'Significant investments in artificial intelligence...',
                'relevance_score': 0.88,
                'source': 'Press Release'
            }
        ],
        'fundamental_summary': 'Demonstrates strong financial health with consistent revenue growth, solid profit margins, and a strong balance sheet.'
    }
    
    return fundamental_data, technical_analysis, economic_context, rag_insights


def main():
    """Main function to test the professional format"""
    
    print("📋 PROFESSIONAL INVESTMENT RECOMMENDATION FORMAT TEST")
    print("=" * 60)
    
    try:
        # Create sample data
        print("\n📊 Creating sample data...")
        fundamental_data, technical_analysis, economic_context, rag_insights = create_sample_data()
        
        # Initialize recommendation engine
        print("🔧 Initializing recommendation engine...")
        engine = RecommendationEngine()
        
        # Generate recommendations
        print("\n🎯 Generating recommendations...")
        recommendations = engine.generate_recommendations(
            fundamental_data=fundamental_data,
            technical_analysis=technical_analysis,
            economic_context=economic_context,
            rag_insights=rag_insights
        )
        
        # Get recommendation summary
        recommendation_summary = engine.get_recommendation_summary(recommendations)
        
        # Initialize LLM analyzer
        print("🤖 Initializing LLM analyzer...")
        llm_analyzer = LLMAnalyzer()
        
        # Prepare complete data structure
        complete_data = {
            'current_price': fundamental_data['current_price'],
            'market_cap': fundamental_data['market_cap'],
            'pe_ratio': fundamental_data['pe_ratio'],
            'technical_indicators': {
                'rsi': f"{technical_analysis['rsi']['current']:.1f}",
                'price_vs_sma20': f"{technical_analysis['moving_averages']['price_vs_sma20']:.1f}%",
                'z_score': f"{technical_analysis['advanced_statistics']['z_score_analysis']['price_z_score']:.2f}",
                'macd_signal': technical_analysis['macd']['signal']
            },
            'economic_context': {
                'sentiment': economic_context['economic_indicators']['economic_sentiment']['overall_sentiment'],
                'fed_policy': economic_context['economic_indicators']['fed_rate']['change_direction'],
                'volatility': economic_context['economic_indicators']['market_volatility']['volatility_regime'],
                'inflation_trend': economic_context['economic_indicators']['inflation']['trend']
            },
            'rag_insights': rag_insights
        }
        
        # Generate professional recommendation report
        print("\n📝 Generating professional recommendation report...")
        professional_report = llm_analyzer.generate_recommendation_report(
            ticker="AAPL",
            recommendations=recommendation_summary,
            data=complete_data
        )
        
        # Display the professional report
        print("\n" + "=" * 60)
        print("📋 PROFESSIONAL INVESTMENT RECOMMENDATION REPORT")
        print("=" * 60)
        print(professional_report)
        
        print("\n✅ PROFESSIONAL FORMAT TEST COMPLETE!")
        print("   The system successfully generated a professional, actionable")
        print("   investment recommendation in institutional-quality format.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during professional format test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 