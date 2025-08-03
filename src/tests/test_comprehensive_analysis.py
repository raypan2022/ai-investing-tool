#!/usr/bin/env python3
"""
Comprehensive Analysis Test - Demonstrating Complete RAG-Based Stock Analysis
This shows what the final LLM input would look like with all components integrated.
"""

import json
import pandas as pd
from datetime import datetime
import sys
import os

from src.data.stock_data import StockDataFetcher
from src.data.sec_filings import SECFilingsFetcher
from src.data.economic_news import EconomicNewsFetcher
from src.rag.vector_store import FAISSVectorStore
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.config import Config

def test_comprehensive_analysis():
    """Test comprehensive analysis combining all components"""
    
    print("🧪 Testing Comprehensive RAG-Based Stock Analysis")
    print("=" * 60)
    
    # Initialize components
    config = Config()
    stock_fetcher = StockDataFetcher(config)
    filings_fetcher = SECFilingsFetcher(config)
    economic_fetcher = EconomicNewsFetcher(config)
    vector_store = FAISSVectorStore(config)
    technical_analyzer = TechnicalAnalyzer(config)
    
    # Test with AAPL
    ticker = "AAPL"
    
    print(f"\n📊 Comprehensive Analysis for {ticker}")
    print("-" * 40)
    
    # 1. Fetch Stock Data
    print("1️⃣ Fetching stock data...")
    stock_info = stock_fetcher.get_stock_info(ticker)
    stock_data = stock_fetcher.get_stock_history(ticker, period="6mo")
    if stock_data is None or stock_data.empty:
        print(f"❌ Failed to fetch stock data for {ticker}")
        return
    
    print(f"✅ Retrieved {len(stock_data)} data points")
    
    # 2. Fetch SEC Filings (RAG Data)
    print("\n2️⃣ Fetching SEC filings and news...")
    filings_data = filings_fetcher.get_latest_10k(ticker)
    if filings_data:
        print(f"✅ Retrieved filing data")
        # Convert to chunks for RAG
        filing_chunks = [
            {
                "content": filings_data.get('text', '')[:1000] + "...",
                "metadata": {"filing_type": "10-K", "date": filings_data.get('filing_date', '2024-01-15')}
            }
        ]
    else:
        print("⚠️ Using mock filing data for demonstration")
        filing_chunks = [
            {
                "content": f"Apple Inc. reported strong quarterly results with revenue growth of 8% year-over-year. iPhone sales increased by 12% while services revenue grew by 15%. The company continues to invest heavily in AI and machine learning technologies.",
                "metadata": {"filing_type": "10-K", "date": "2024-01-15"}
            },
            {
                "content": f"Apple's supply chain optimization and cost management initiatives have improved gross margins to 45.2%. The company expects continued growth in emerging markets and strong demand for premium products.",
                "metadata": {"filing_type": "10-K", "date": "2024-01-15"}
            }
        ]
    
    # 3. Perform Technical Analysis
    print("\n3️⃣ Performing technical analysis...")
    technical_analysis = technical_analyzer.analyze_stock(ticker, stock_data)
    if 'error' in technical_analysis:
        print(f"❌ Technical analysis failed: {technical_analysis['error']}")
        return
    
    print("✅ Technical analysis completed")
    
    # 4. Simulate RAG Search (Vector Search)
    print("\n4️⃣ Performing RAG search...")
    
    # Simulate relevant document retrieval
    rag_results = {
        "relevant_documents": [
            {
                "content": "Apple's AI strategy focuses on on-device processing and privacy. The company has made significant investments in machine learning capabilities across its product ecosystem.",
                "relevance_score": 0.89,
                "source": "10-K Filing",
                "date": "2024-01-15"
            },
            {
                "content": "iPhone sales in emerging markets grew 25% year-over-year, with particularly strong performance in India and Brazil. Services revenue continues to be the fastest-growing segment.",
                "relevance_score": 0.85,
                "source": "10-K Filing", 
                "date": "2024-01-15"
            },
            {
                "content": "Apple's supply chain diversification efforts have reduced dependency on single suppliers. The company expects improved gross margins through 2024 due to cost optimization.",
                "relevance_score": 0.82,
                "source": "10-K Filing",
                "date": "2024-01-15"
            }
        ],
        "market_sentiment": "positive",
        "key_themes": ["AI investment", "emerging markets growth", "supply chain optimization", "services expansion"],
        "fundamental_summary": "Apple shows strong fundamentals with growing services revenue, AI investments, and emerging market expansion. Supply chain optimization is improving margins."
    }
    
    print("✅ RAG search completed")
    
    # 5. Fetch Economic and Political Data
    print("\n5️⃣ Fetching economic and political data...")
    economic_data = economic_fetcher.get_economic_indicators()
    political_data = economic_fetcher.get_political_events()
    print("✅ Economic and political data retrieved")
    
    # 6. Compile Comprehensive Analysis
    print("\n6️⃣ Compiling comprehensive analysis...")
    
    comprehensive_analysis = {
        "analysis_metadata": {
            "ticker": ticker,
            "analysis_date": datetime.now().isoformat(),
            "analysis_type": "comprehensive_rag_technical",
            "components_used": ["stock_data", "sec_filings", "rag_search", "technical_analysis"]
        },
        
        "fundamental_data": {
            "current_price": stock_data['Close'].iloc[-1],
            "market_cap": stock_info.get('market_cap', 0),
            "pe_ratio": stock_info.get('pe_ratio', 0),
            "dividend_yield": stock_info.get('dividend_yield', 0),
            "beta": stock_info.get('beta', 1.0),
            "price_changes": {
                "1d": technical_analysis.get('price_change_1d', 0),
                "5d": technical_analysis.get('price_change_5d', 0),
                "20d": technical_analysis.get('price_change_20d', 0)
            }
        },
        
        "rag_insights": {
            "relevant_documents": rag_results["relevant_documents"],
            "market_sentiment": rag_results["market_sentiment"],
            "key_themes": rag_results["key_themes"],
            "fundamental_summary": "Apple shows strong fundamentals with growing services revenue, AI investments, and emerging market expansion. Supply chain optimization is improving margins."
        },
        
        "economic_context": {
            "economic_indicators": economic_data,
            "political_events": political_data,
            "market_environment": {
                "fed_policy": economic_data.get('fed_rate', {}).get('change_direction', 'hold'),
                "economic_sentiment": economic_data.get('economic_sentiment', {}).get('overall_sentiment', 'neutral'),
                "volatility_regime": economic_data.get('market_volatility', {}).get('volatility_regime', 'moderate'),
                "yield_curve": economic_data.get('treasury_yields', {}).get('curve_inverted', False),
                "political_uncertainty": political_data.get('election_events', {}).get('uncertainty_level', 'moderate')
            }
        },
        
        "technical_analysis": {
            "overall_signal": technical_analysis.get('overall_signal', 'HOLD'),
            "confidence": technical_analysis.get('confidence', 50),
            "key_indicators": {
                "rsi": technical_analysis.get('rsi', {}).get('current', 50),
                "macd_signal": technical_analysis.get('macd', {}).get('signal', 'neutral'),
                "price_vs_sma20": technical_analysis.get('moving_averages', {}).get('price_vs_sma_20', 'neutral'),
                "bollinger_position": technical_analysis.get('bollinger_bands', {}).get('position', 'middle')
            },
            "support_resistance": {
                "support_1": technical_analysis.get('pivot_points', {}).get('support_1', 0),
                "resistance_1": technical_analysis.get('pivot_points', {}).get('resistance_1', 0)
            },
            "advanced_statistics": technical_analysis.get('advanced_statistics', {}),
            "investment_strategy": technical_analysis.get('investment_strategy', {})
        },
        
        "risk_assessment": {
            "technical_risk": "moderate" if technical_analysis.get('confidence', 50) > 70 else "high",
            "fundamental_risk": "low" if rag_results["market_sentiment"] == "positive" else "moderate",
            "volatility_risk": technical_analysis.get('advanced_statistics', {}).get('garch_model', {}).get('risk_level', 'medium'),
            "overall_risk": "moderate"
        },
        
        "llm_recommendation_prompt": {
            "context": f"""
You are analyzing {ticker} stock with comprehensive data from multiple sources:

FUNDAMENTAL CONTEXT:
- Current Price: ${technical_analysis.get('current_price', 0):.2f}
- Market Sentiment: {rag_results["market_sentiment"]}
- Key Themes: {', '.join(rag_results["key_themes"])}
- Fundamental Summary: {rag_results["fundamental_summary"]}

ECONOMIC & POLITICAL CONTEXT:
- Fed Policy: {economic_data.get('fed_rate', {}).get('change_direction', 'hold')} (Rate: {economic_data.get('fed_rate', {}).get('current_rate', 0)}%)
- Economic Sentiment: {economic_data.get('economic_sentiment', {}).get('overall_sentiment', 'neutral')}
- Market Volatility: {economic_data.get('market_volatility', {}).get('volatility_regime', 'moderate')} (VIX: {economic_data.get('market_volatility', {}).get('current_vix', 0):.1f})
- Yield Curve: {'Inverted' if economic_data.get('treasury_yields', {}).get('curve_inverted', False) else 'Normal'}
- Political Uncertainty: {political_data.get('election_events', {}).get('uncertainty_level', 'moderate')}
- Key Economic Factors: {', '.join(economic_data.get('economic_sentiment', {}).get('key_factors', ['none']))}

TECHNICAL CONTEXT:
- Overall Signal: {technical_analysis.get('overall_signal', 'HOLD')} ({technical_analysis.get('confidence', 50):.1f}% confidence)
- RSI: {technical_analysis.get('rsi', {}).get('current', 50):.1f}
- Price vs SMA20: {technical_analysis.get('moving_averages', {}).get('price_vs_sma_20', 'neutral')}
- Investment Strategy: {technical_analysis.get('investment_strategy', {}).get('recommended_timeframe', 'medium_term')} term, {technical_analysis.get('investment_strategy', {}).get('primary_driver', 'balanced')} approach

RISK CONTEXT:
- Technical Risk: {technical_analysis.get('advanced_statistics', {}).get('garch_model', {}).get('risk_level', 'medium')}
- Economic Risk: {economic_data.get('economic_sentiment', {}).get('market_implications', {}).get('risk_level', 'moderate')}
- Political Risk: {political_data.get('market_impact', {}).get('overall_impact', 'neutral')}
- Support: ${technical_analysis.get('pivot_points', {}).get('support_1', 0):.2f}
- Resistance: ${technical_analysis.get('pivot_points', {}).get('resistance_1', 0):.2f}

Based on this comprehensive analysis including economic and political factors, provide a detailed BUY/HOLD/SELL recommendation with:
1. Clear reasoning for your recommendation
2. Specific entry/exit price targets
3. Risk management guidelines considering economic/political risks
4. Expected timeframe for the position
5. Key factors that could change your recommendation (including economic/political events)
6. How current economic conditions affect this specific stock
""",
            "expected_format": "structured_recommendation"
        }
    }
    
    # Save comprehensive analysis
    output_file = f"comprehensive_analysis_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(comprehensive_analysis, f, indent=2, default=str)
    
    print(f"✅ Comprehensive analysis saved to: {output_file}")
    
    # Display summary
    print(f"\n📋 Comprehensive Analysis Summary for {ticker}")
    print("=" * 50)
    print(f"💰 Current Price: ${comprehensive_analysis['fundamental_data']['current_price']:.2f}")
    print(f"📈 Technical Signal: {comprehensive_analysis['technical_analysis']['overall_signal']} ({comprehensive_analysis['technical_analysis']['confidence']:.1f}% confidence)")
    print(f"📊 Market Sentiment: {comprehensive_analysis['rag_insights']['market_sentiment']}")
    print(f"🏛️ Economic Sentiment: {comprehensive_analysis['economic_context']['economic_indicators'].get('economic_sentiment', {}).get('overall_sentiment', 'neutral')}")
    print(f"📈 Fed Policy: {comprehensive_analysis['economic_context']['economic_indicators'].get('fed_rate', {}).get('change_direction', 'hold')} ({comprehensive_analysis['economic_context']['economic_indicators'].get('fed_rate', {}).get('current_rate', 0)}%)")
    print(f"🎯 Investment Strategy: {comprehensive_analysis['technical_analysis']['investment_strategy'].get('recommended_timeframe', 'medium_term')} term, {comprehensive_analysis['technical_analysis']['investment_strategy'].get('primary_driver', 'balanced')} approach")
    print(f"⚠️ Overall Risk: {comprehensive_analysis['risk_assessment']['overall_risk']}")
    print(f"🔍 Key Themes: {', '.join(comprehensive_analysis['rag_insights']['key_themes'])}")
    print(f"🏛️ Key Economic Factors: {', '.join(comprehensive_analysis['economic_context']['economic_indicators'].get('economic_sentiment', {}).get('key_factors', ['none']))}")
    
    print(f"\n📄 Detailed analysis saved to: {output_file}")
    print("\n🎯 This is what the LLM will receive for final recommendation!")
    
    return comprehensive_analysis

if __name__ == "__main__":
    test_comprehensive_analysis() 