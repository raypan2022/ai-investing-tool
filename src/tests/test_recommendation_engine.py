#!/usr/bin/env python3
"""
Test script for the Recommendation Engine

This script demonstrates the recommendation engine with sample data
to show how it generates both short-term and long-term recommendations.
"""

import sys
import os

from src.analysis.recommendation_engine import RecommendationEngine


def create_sample_data():
    """Create sample data for testing the recommendation engine"""
    
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
    """Main function to test the recommendation engine"""
    
    print("🧠 RECOMMENDATION ENGINE TEST")
    print("=" * 40)
    
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
        
        # Display results
        print("\n" + "=" * 40)
        print("📋 RECOMMENDATION RESULTS")
        print("=" * 40)
        
        # Short-term analysis
        short_term = recommendations['short_term']
        print(f"\n📈 SHORT-TERM OUTLOOK (1-2 weeks):")
        print(f"   Recommendation: {short_term.recommendation.value}")
        print(f"   Confidence: {short_term.confidence.value}")
        print(f"   Score: {short_term.score:.1f}/100")
        print(f"   Factor Scores:")
        print(f"     • Technical: {short_term.factor_scores.technical:.1f}/100")
        print(f"     • Fundamental: {short_term.factor_scores.fundamental:.1f}/100")
        print(f"     • Economic: {short_term.factor_scores.economic:.1f}/100")
        print(f"     • News Sentiment: {short_term.factor_scores.news_sentiment:.1f}/100")
        
        # Long-term analysis
        long_term = recommendations['long_term']
        print(f"\n📊 LONG-TERM OUTLOOK (6-12 months):")
        print(f"   Recommendation: {long_term.recommendation.value}")
        print(f"   Confidence: {long_term.confidence.value}")
        print(f"   Score: {long_term.score:.1f}/100")
        print(f"   Factor Scores:")
        print(f"     • Fundamental: {long_term.factor_scores.fundamental:.1f}/100")
        print(f"     • News Sentiment: {long_term.factor_scores.news_sentiment:.1f}/100")
        print(f"     • Technical: {long_term.factor_scores.technical:.1f}/100")
        print(f"     • Economic: {long_term.factor_scores.economic:.1f}/100")
        
        # Get recommendation summary
        print(f"\n📝 RECOMMENDATION SUMMARY:")
        summary = engine.get_recommendation_summary(recommendations)
        print(f"   Overall Sentiment: {summary['summary']['overall_sentiment']}")
        print(f"   Confidence Agreement: {summary['summary']['confidence_agreement']}")
        print(f"   Factor Consistency:")
        for factor, consistency in summary['summary']['factor_consistency'].items():
            print(f"     • {factor.title()}: {consistency:.1f}")
        
        # Show weight configurations
        print(f"\n⚖️  WEIGHT CONFIGURATIONS:")
        print(f"   Short-term weights:")
        for factor, weight in engine.SHORT_TERM_WEIGHTS.items():
            print(f"     • {factor.title()}: {weight*100:.0f}%")
        print(f"   Long-term weights:")
        for factor, weight in engine.LONG_TERM_WEIGHTS.items():
            print(f"     • {factor.title()}: {weight*100:.0f}%")
        
        print(f"\n✅ RECOMMENDATION ENGINE TEST COMPLETE!")
        print(f"   The engine successfully generated both short-term and long-term")
        print(f"   recommendations with confidence levels and factor breakdowns.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during recommendation engine test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 