#!/usr/bin/env python3
"""
Complete RAG-Based Stock Analysis Test

This script demonstrates the complete workflow combining:
1. Stock data fetching
2. SEC filings processing
3. Technical analysis
4. Economic context
5. Recommendation engine with both short-term and long-term outlooks
"""

import sys
import os
import json
from typing import Dict, Any

from src.data.stock_data import StockDataFetcher
from src.data.sec_filings import SECFilingsFetcher
from src.data.economic_news import EconomicNewsFetcher
from src.rag.vector_store import FAISSVectorStore
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.analysis.recommendation_engine import RecommendationEngine
from src.analysis.llm_analyzer import LLMAnalyzer
import src.config as config


def create_mock_rag_results(ticker: str) -> Dict[str, Any]:
    """Create mock RAG results for demonstration"""
    return {
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
                'content': f'{ticker} reported strong quarterly results with revenue growth of 15%...',
                'relevance_score': 0.92,
                'source': 'SEC Filing'
            },
            {
                'title': 'AI Strategy Update',
                'content': f'{ticker} announced significant investments in artificial intelligence...',
                'relevance_score': 0.88,
                'source': 'Press Release'
            },
            {
                'title': 'Market Analysis',
                'content': f'Analysts remain bullish on {ticker} due to strong fundamentals...',
                'relevance_score': 0.85,
                'source': 'Research Report'
            }
        ],
        'fundamental_summary': f'{ticker} demonstrates strong financial health with consistent revenue growth, solid profit margins, and a strong balance sheet. The company is well-positioned in its market with competitive advantages and continued investment in innovation.'
    }


def format_llm_prompt(ticker: str, recommendations: Dict[str, Any], data: Dict[str, Any]) -> str:
    """Generate comprehensive LLM prompt with all analysis data"""
    
    return f"""
You are a professional investment analyst. Provide comprehensive recommendations for {ticker} using all available data sources.

COMPLETE ANALYSIS DATA:
- Current Price: ${data['current_price']}
- Market Cap: ${data['market_cap']:,.0f}
- PE Ratio: {data['pe_ratio']}

SHORT-TERM ANALYSIS (1-2 weeks):
- Recommendation: {recommendations['short_term']['recommendation']}
- Confidence: {recommendations['short_term']['confidence']}
- Score: {recommendations['short_term']['score']:.1f}/100
- Factor Scores: 
  • Technical: {recommendations['short_term']['factor_scores']['technical']:.1f}/100
  • Fundamental: {recommendations['short_term']['factor_scores']['fundamental']:.1f}/100
  • Economic: {recommendations['short_term']['factor_scores']['economic']:.1f}/100
  • News Sentiment: {recommendations['short_term']['factor_scores']['news_sentiment']:.1f}/100

LONG-TERM ANALYSIS (6-12 months):
- Recommendation: {recommendations['long_term']['recommendation']}
- Confidence: {recommendations['long_term']['confidence']}
- Score: {recommendations['long_term']['score']:.1f}/100
- Factor Scores:
  • Fundamental: {recommendations['long_term']['factor_scores']['fundamental']:.1f}/100
  • News Sentiment: {recommendations['long_term']['factor_scores']['news_sentiment']:.1f}/100
  • Technical: {recommendations['long_term']['factor_scores']['technical']:.1f}/100
  • Economic: {recommendations['long_term']['factor_scores']['economic']:.1f}/100

TECHNICAL INDICATORS:
- RSI: {data['technical_indicators']['rsi']}
- Price vs SMA20: {data['technical_indicators']['price_vs_sma20']}
- Z-Score: {data['technical_indicators']['z_score']}
- MACD Signal: {data['technical_indicators']['macd_signal']}

ECONOMIC CONTEXT:
- Economic Sentiment: {data['economic_context']['sentiment']}
- Fed Policy: {data['economic_context']['fed_policy']}
- Market Volatility: {data['economic_context']['volatility']}
- Inflation Trend: {data['economic_context']['inflation_trend']}

NEWS & FILINGS INSIGHTS:
- Market Sentiment: {data['rag_insights']['market_sentiment']}
- Key Themes: {', '.join(data['rag_insights']['key_themes'])}
- Relevant Documents: {len(data['rag_insights']['relevant_documents'])} high-relevance documents
- Fundamental Summary: {data['rag_insights']['fundamental_summary']}

Provide your analysis in this EXACT format:

SHORT-TERM OUTLOOK (1-2 weeks):
RECOMMENDATION: [BUY/HOLD/SELL]
CONFIDENCE: [High/Moderate/Low]

SUMMARY: [One sentence summary incorporating all factors]

EVIDENCE:
• [Technical factor with specific data]
• [Fundamental factor with specific data]
• [Economic factor with specific data]
• [News/Filings factor with specific data]

ACTION PLAN:
• Entry: $[price]
• Target: $[price] ([percentage])
• Stop Loss: $[price] ([percentage])

LONG-TERM OUTLOOK (6-12 months):
RECOMMENDATION: [BUY/HOLD/SELL]
CONFIDENCE: [High/Moderate/Low]

SUMMARY: [One sentence summary incorporating all factors]

EVIDENCE:
• [Fundamental factor with specific data]
• [News/Filings factor with specific data]
• [Technical factor with specific data]
• [Economic factor with specific data]

ACTION PLAN:
• Entry Strategy: [Specific guidance based on all factors]
• Target: $[price] ([percentage])
• Hold Period: [X months]

KEY RISKS:
• [Primary risk factor]
• [Secondary risk factor]

MONITORING:
• [Factor to watch for short-term]
• [Factor to watch for long-term]
"""


def main():
    """Main function to demonstrate complete analysis workflow"""
    
    print("🚀 COMPLETE RAG-BASED STOCK ANALYSIS")
    print("=" * 50)
    
    # Initialize all components
    ticker = "AAPL"
    
    print(f"\n📊 Analyzing {ticker}...")
    
    try:
        # 1. Fetch stock data
        print("\n1️⃣ Fetching stock data...")
        stock_fetcher = StockDataFetcher(config)
        stock_info = stock_fetcher.get_stock_info(ticker)
        stock_history = stock_fetcher.get_stock_history(ticker, period="6mo")
        
        print(f"   ✅ Current Price: ${stock_info['current_price']}")
        print(f"   ✅ Market Cap: ${stock_info['market_cap']:,.0f}")
        print(f"   ✅ PE Ratio: {stock_info['pe_ratio']}")
        
        # 2. Fetch SEC filings
        print("\n2️⃣ Fetching SEC filings...")
        filings_fetcher = SECFilingsFetcher(config)
        filings_data = filings_fetcher.get_latest_10k(ticker)
        
        # Create filing chunks for RAG
        filing_chunks = [
            {
                'content': filings_data['text'][:1000],  # First 1000 chars
                'metadata': {'source': '10-K', 'ticker': ticker}
            }
        ]
        
        print(f"   ✅ Retrieved {len(filing_chunks)} filing chunks")
        
        # 3. Initialize vector store (mock for demo)
        print("\n3️⃣ Setting up vector store...")
        vector_store = FAISSVectorStore(config)
        print("   ✅ Vector store initialized")
        
        # 4. Perform technical analysis
        print("\n4️⃣ Performing technical analysis...")
        technical_analyzer = TechnicalAnalyzer(config)
        technical_analysis = technical_analyzer.analyze_stock(ticker, stock_history)
        
        print(f"   ✅ Overall Signal: {technical_analysis['overall_signal']}")
        print(f"   ✅ Confidence: {technical_analysis['confidence']}")
        
        # 5. Fetch economic context
        print("\n5️⃣ Fetching economic context...")
        economic_fetcher = EconomicNewsFetcher(config)
        economic_context = economic_fetcher.get_economic_indicators()
        political_events = economic_fetcher.get_political_events()
        
        print(f"   ✅ Economic Sentiment: {economic_context['economic_sentiment']['overall_sentiment']}")
        print(f"   ✅ Fed Policy: {economic_context['fed_rate']['change_direction']}")
        
        # 6. Create RAG insights (mock for demo)
        print("\n6️⃣ Generating RAG insights...")
        rag_insights = create_mock_rag_results(ticker)
        
        print(f"   ✅ Market Sentiment: {rag_insights['market_sentiment']}")
        print(f"   ✅ Key Themes: {len(rag_insights['key_themes'])} identified")
        
        # 7. Generate recommendations
        print("\n7️⃣ Generating recommendations...")
        recommendation_engine = RecommendationEngine()
        
        recommendations = recommendation_engine.generate_recommendations(
            fundamental_data=stock_info,
            technical_analysis=technical_analysis,
            economic_context=economic_context,
            rag_insights=rag_insights
        )
        
        # Get recommendation summary
        recommendation_summary = recommendation_engine.get_recommendation_summary(recommendations)
        
        print(f"   ✅ Short-term: {recommendations['short_term'].recommendation.value} ({recommendations['short_term'].confidence.value} confidence)")
        print(f"   ✅ Long-term: {recommendations['long_term'].recommendation.value} ({recommendations['long_term'].confidence.value} confidence)")
        
        # 8. Prepare data for LLM
        print("\n8️⃣ Preparing LLM input...")
        
        # Extract key technical indicators with safe access
        technical_indicators = {
            'rsi': f"{technical_analysis.get('rsi', {}).get('current', 50.0):.1f}",
            'price_vs_sma20': f"{technical_analysis.get('moving_averages', {}).get('price_vs_sma20', 0.0):.1f}%",
            'z_score': f"{technical_analysis.get('advanced_statistics', {}).get('z_score_analysis', {}).get('price_z_score', 0.0):.2f}",
            'macd_signal': technical_analysis.get('macd', {}).get('signal', 'neutral')
        }
        
        # Extract economic context with safe access
        economic_context_summary = {
            'sentiment': economic_context.get('economic_sentiment', {}).get('overall_sentiment', 'neutral'),
            'fed_policy': economic_context.get('fed_rate', {}).get('change_direction', 'neutral'),
            'volatility': economic_context.get('market_volatility', {}).get('volatility_regime', 'normal'),
            'inflation_trend': economic_context.get('inflation', {}).get('trend', 'stable')
        }
        
        # Prepare complete data structure
        complete_data = {
            'current_price': stock_info['current_price'],
            'market_cap': stock_info['market_cap'],
            'pe_ratio': stock_info['pe_ratio'],
            'technical_indicators': technical_indicators,
            'economic_context': economic_context_summary,
            'rag_insights': rag_insights
        }
        
        # Generate professional recommendation report
        llm_analyzer = LLMAnalyzer()
        professional_report = llm_analyzer.generate_recommendation_report(
            ticker=ticker,
            recommendations=recommendation_summary,
            data=complete_data
        )
        
        print("   ✅ Professional recommendation report generated")
        
        # 9. Display results
        print("\n" + "=" * 50)
        print("📋 COMPLETE ANALYSIS RESULTS")
        print("=" * 50)
        
        print(f"\n🎯 {ticker} ANALYSIS SUMMARY:")
        print(f"   Current Price: ${stock_info['current_price']}")
        print(f"   Market Cap: ${stock_info['market_cap']:,.0f}")
        print(f"   PE Ratio: {stock_info['pe_ratio']}")
        
        print(f"\n📈 SHORT-TERM OUTLOOK (1-2 weeks):")
        short_term = recommendations['short_term']
        print(f"   Recommendation: {short_term.recommendation.value}")
        print(f"   Confidence: {short_term.confidence.value}")
        print(f"   Score: {short_term.score:.1f}/100")
        print(f"   Factor Scores:")
        print(f"     • Technical: {short_term.factor_scores.technical:.1f}/100")
        print(f"     • Fundamental: {short_term.factor_scores.fundamental:.1f}/100")
        print(f"     • Economic: {short_term.factor_scores.economic:.1f}/100")
        print(f"     • News Sentiment: {short_term.factor_scores.news_sentiment:.1f}/100")
        
        print(f"\n📊 LONG-TERM OUTLOOK (6-12 months):")
        long_term = recommendations['long_term']
        print(f"   Recommendation: {long_term.recommendation.value}")
        print(f"   Confidence: {long_term.confidence.value}")
        print(f"   Score: {long_term.score:.1f}/100")
        print(f"   Factor Scores:")
        print(f"     • Fundamental: {long_term.factor_scores.fundamental:.1f}/100")
        print(f"     • News Sentiment: {long_term.factor_scores.news_sentiment:.1f}/100")
        print(f"     • Technical: {long_term.factor_scores.technical:.1f}/100")
        print(f"     • Economic: {long_term.factor_scores.economic:.1f}/100")
        
        print(f"\n🔍 TECHNICAL INDICATORS:")
        print(f"   RSI: {technical_indicators['rsi']}")
        print(f"   Price vs SMA20: {technical_indicators['price_vs_sma20']}")
        print(f"   Z-Score: {technical_indicators['z_score']}")
        print(f"   MACD Signal: {technical_indicators['macd_signal']}")
        
        print(f"\n🌍 ECONOMIC CONTEXT:")
        print(f"   Economic Sentiment: {economic_context_summary['sentiment']}")
        print(f"   Fed Policy: {economic_context_summary['fed_policy']}")
        print(f"   Market Volatility: {economic_context_summary['volatility']}")
        print(f"   Inflation Trend: {economic_context_summary['inflation_trend']}")
        
        print(f"\n📰 NEWS & FILINGS INSIGHTS:")
        print(f"   Market Sentiment: {rag_insights['market_sentiment']}")
        print(f"   Key Themes: {', '.join(rag_insights['key_themes'][:3])}...")
        print(f"   Relevant Documents: {len(rag_insights['relevant_documents'])} high-relevance documents")
        
        print(f"\n📋 PROFESSIONAL RECOMMENDATION REPORT:")
        print("   " + "-" * 40)
        print(professional_report)
        print("   " + "-" * 40)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"   The system has successfully combined all data sources:")
        print(f"   • Stock fundamentals and technical indicators")
        print(f"   • Economic context and market conditions")
        print(f"   • News sentiment and SEC filings analysis")
        print(f"   • Multi-factor recommendation scoring")
        print(f"   • Both short-term and long-term outlooks")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 