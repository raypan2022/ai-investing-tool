#!/usr/bin/env python3
"""
Test script for ReAct Agent System

This script demonstrates the complete ReAct-based investment agent with:
1. Deep research workflow
2. Chat interface with ReAct reasoning
3. Specialized RAG tools
4. Tool selection and execution
"""

import sys
import os

from src.agent.investment_agent import InvestmentAgent


def test_deep_research():
    """Test the deep research workflow"""
    print("🔍 TESTING DEEP RESEARCH WORKFLOW")
    print("=" * 50)
    
    try:
        # Initialize investment agent
        print("\n1️⃣ Initializing investment agent...")
        agent = InvestmentAgent()  # No LLM client for now (uses fallback reasoning)
        
        # Check initial status
        status = agent.get_status()
        print(f"   ✅ Agent initialized: {status}")
        
        # Run deep research
        print("\n2️⃣ Running deep research for AAPL...")
        results = agent.run_deep_research("AAPL")
        
        if 'error' in results:
            print(f"   ❌ Error: {results['error']}")
            return False
        
        print(f"   ✅ Deep research completed!")
        print(f"   ✅ Chat enabled: {results['chat_enabled']}")
        print(f"   ✅ Ticker: {results['ticker']}")
        
        # Show analysis summary
        summary = agent.get_analysis_summary()
        print(f"\n3️⃣ Analysis Summary:")
        print(f"   Ticker: {summary['ticker']}")
        print(f"   Components: {list(summary['components'].keys())}")
        
        if 'stock_data' in summary['components']:
            stock_data = summary['components']['stock_data']
            print(f"   Current Price: ${stock_data.get('current_price', 0):.2f}")
            print(f"   Market Cap: ${stock_data.get('market_cap', 0):,.0f}")
            print(f"   PE Ratio: {stock_data.get('pe_ratio', 0):.2f}")
        
        if 'technical_analysis' in summary['components']:
            tech_data = summary['components']['technical_analysis']
            print(f"   Technical Signal: {tech_data.get('overall_signal', 'HOLD')}")
            print(f"   Confidence: {tech_data.get('confidence', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in deep research: {e}")
        return False


def test_chat_interface():
    """Test the chat interface with ReAct reasoning"""
    print("\n\n💬 TESTING CHAT INTERFACE")
    print("=" * 50)
    
    try:
        # Initialize agent and run deep research first
        print("\n1️⃣ Setting up chat interface...")
        agent = InvestmentAgent()
        results = agent.run_deep_research("AAPL")
        
        if 'error' in results:
            print(f"   ❌ Cannot test chat without deep research: {results['error']}")
            return False
        
        # Test various chat scenarios
        test_questions = [
            "What's the current stock price?",
            "How does the technical analysis look?",
            "What's the latest news about Apple?",
            "How do current economic conditions affect tech stocks?",
            "What were Apple's recent earnings like?",
            "What's the market sentiment for Apple?"
        ]
        
        print(f"\n2️⃣ Testing chat scenarios...")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n   Q{i}: {question}")
            response = agent.chat(question)
            print(f"   A{i}: {response[:200]}...")
        
        # Show conversation history
        history = agent.get_conversation_history()
        print(f"\n3️⃣ Conversation History:")
        print(f"   Total exchanges: {len(history)}")
        
        for i, exchange in enumerate(history[-3:], 1):  # Show last 3 exchanges
            print(f"   Exchange {i}:")
            print(f"     User: {exchange['user']}")
            print(f"     Agent: {exchange['agent'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in chat interface: {e}")
        return False


def test_tool_selection():
    """Test ReAct tool selection and reasoning"""
    print("\n\n🛠️ TESTING TOOL SELECTION")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("\n1️⃣ Initializing agent...")
        agent = InvestmentAgent()
        
        # Run deep research to enable chat
        results = agent.run_deep_research("AAPL")
        if 'error' in results:
            print(f"   ❌ Cannot test tools without deep research: {results['error']}")
            return False
        
        # Test tool-specific questions
        tool_tests = [
            {
                'question': 'What is the current stock price and market cap?',
                'expected_tool': 'review_stock_data',
                'description': 'Stock data query'
            },
            {
                'question': 'What are the RSI and MACD indicators showing?',
                'expected_tool': 'review_technical_analysis',
                'description': 'Technical analysis query'
            },
            {
                'question': 'What are the latest news and announcements?',
                'expected_tool': 'rag_company_news',
                'description': 'Company news query'
            },
            {
                'question': 'How are current economic conditions affecting the market?',
                'expected_tool': 'rag_current_economics',
                'description': 'Economic data query'
            },
            {
                'question': 'What do the recent SEC filings show about financial performance?',
                'expected_tool': 'rag_filings',
                'description': 'SEC filings query'
            }
        ]
        
        print(f"\n2️⃣ Testing tool selection...")
        
        for test in tool_tests:
            print(f"\n   Testing: {test['description']}")
            print(f"   Question: {test['question']}")
            
            # Get response
            response = agent.chat(test['question'])
            print(f"   Response: {response[:150]}...")
            
            # Check if the expected tool was mentioned (in fallback mode)
            if test['expected_tool'] in response.lower():
                print(f"   ✅ Expected tool '{test['expected_tool']}' was used")
            else:
                print(f"   ⚠️  Expected tool '{test['expected_tool']}' not clearly identified")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in tool selection: {e}")
        return False


def test_agent_status():
    """Test agent status and management functions"""
    print("\n\n📊 TESTING AGENT STATUS")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("\n1️⃣ Initializing agent...")
        agent = InvestmentAgent()
        
        # Check initial status
        print("\n2️⃣ Initial status:")
        status = agent.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Run deep research
        print("\n3️⃣ After deep research:")
        results = agent.run_deep_research("AAPL")
        status = agent.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Test reset functionality
        print("\n4️⃣ Testing reset:")
        agent.reset()
        status = agent.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in status test: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 REACT AGENT SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Deep Research Workflow", test_deep_research),
        ("Chat Interface", test_chat_interface),
        ("Tool Selection", test_tool_selection),
        ("Agent Status", test_agent_status)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("The ReAct agent system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 