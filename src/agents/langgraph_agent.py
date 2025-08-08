"""LangGraph ReAct agent for financial analysis using FinGPT"""

from typing import Dict, Any, List, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from src.models import get_model_client
from src.tools.tools import TOOLS


class FinancialAnalysisState(TypedDict):
    """State for financial analysis workflow"""
    query: str
    ticker: str
    messages: Annotated[List[AnyMessage], operator.add]
    analysis_result: str


class FinancialReActAgent:
    """LangGraph ReAct agent for financial analysis"""
    
    def __init__(self, model_type: str = "fingpt", **model_kwargs):
        """
        Initialize the financial ReAct agent
        
        Args:
            model_type: Type of model to use ('fingpt' or 'openai')
            **model_kwargs: Additional arguments for model client
        """
        self.model_type = model_type
        self.model_client = get_model_client(model_type, **model_kwargs)
        
        # Bind tools to model
        self.llm_with_tools = self.model_client.bind_tools(TOOLS)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph ReAct workflow"""
        
        # Create the graph
        workflow = StateGraph(FinancialAnalysisState)
        
        # Add nodes
        workflow.add_node("reasoner", self._reasoner_node)
        workflow.add_node("tools", ToolNode(TOOLS))
        
        # Add edges
        workflow.add_edge(START, "reasoner")
        workflow.add_conditional_edges(
            "reasoner",
            tools_condition,  # Routes to tools if tool calls, otherwise END
        )
        workflow.add_edge("tools", "reasoner")
        
        return workflow.compile()
    
    def _reasoner_node(self, state: FinancialAnalysisState) -> Dict[str, Any]:
        """
        Reasoner node that decides whether to use tools or provide final answer
        """
        messages = state["messages"]
        
        # Add system message for financial analysis
        system_msg = SystemMessage(
            content=self._get_system_prompt()
        )
        
        # If no messages yet, add the user query
        if not messages:
            user_msg = HumanMessage(content=state["query"])
            messages = [user_msg]
        
        # Get response from model
        response = self.llm_with_tools.invoke([system_msg] + messages)
        
        return {"messages": [response]}
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for financial analysis"""
        return """You are FinGPT, a specialized AI assistant for financial analysis and investment research.

Your role is to provide comprehensive, data-driven investment analysis using the available tools.

Available tools:
- get_stock_data: Get comprehensive stock information
- get_stock_price: Get current stock price
- calculate_rsi: Calculate RSI technical indicator
- calculate_moving_averages: Calculate moving averages
- get_news_sentiment: Analyze recent news sentiment
- query_financial_documents: Query SEC filings and documents
- calculate_financial_ratios: Calculate financial ratios

Analysis Framework:
1. Gather fundamental data (price, ratios, company info)
2. Perform technical analysis (RSI, moving averages)
3. Analyze sentiment and news
4. Query relevant financial documents if needed
5. Provide comprehensive buy/hold/sell recommendation

Always provide:
- Clear reasoning for your analysis
- Specific data points and evidence
- Risk assessment
- Price targets when appropriate
- Confidence level in your recommendation

Be thorough but efficient in your tool usage."""
    
    def analyze_stock(self, ticker: str, query: str = None) -> Dict[str, Any]:
        """
        Analyze a stock using the ReAct pattern
        
        Args:
            ticker: Stock ticker symbol
            query: Specific analysis query (optional)
        """
        if not query:
            query = f"Provide a comprehensive investment analysis for {ticker}. Should I buy, hold, or sell?"
        
        # Initial state
        initial_state = {
            "query": query,
            "ticker": ticker,
            "messages": [],
            "analysis_result": ""
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract final analysis
        final_message = result["messages"][-1]
        
        return {
            "ticker": ticker,
            "query": query,
            "analysis": final_message.content,
            "messages": result["messages"],
            "tool_calls_made": self._count_tool_calls(result["messages"])
        }
    
    def _count_tool_calls(self, messages: List[AnyMessage]) -> int:
        """Count number of tool calls made during analysis"""
        count = 0
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                count += len(msg.tool_calls)
        return count


# Convenience function for quick analysis
def analyze_stock_with_fingpt(ticker: str, 
                            query: str = None,
                            endpoint_url: str = None,
                            api_key: str = None) -> Dict[str, Any]:
    """
    Quick function to analyze a stock with FinGPT
    
    Args:
        ticker: Stock ticker symbol
        query: Analysis query
        endpoint_url: FinGPT endpoint URL
        api_key: API key for FinGPT
    """
    agent = FinancialReActAgent(
        model_type="fingpt",
        endpoint_url=endpoint_url,
        api_key=api_key
    )
    
    return agent.analyze_stock(ticker, query)


# Test function
def test_financial_agent():
    """Test the financial ReAct agent"""
    # For testing, use OpenAI (since FinGPT endpoint might not be ready)
    agent = FinancialReActAgent(model_type="openai")
    
    result = agent.analyze_stock("AAPL", "Should I invest in Apple stock?")
    
    print("=== Financial ReAct Agent Analysis ===")
    print(f"Ticker: {result['ticker']}")
    print(f"Query: {result['query']}")
    print(f"Tool calls made: {result['tool_calls_made']}")
    print("\nAnalysis:")
    print(result['analysis'])
    
    return result


if __name__ == "__main__":
    test_financial_agent()