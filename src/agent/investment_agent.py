"""
Investment Agent - Complete Chat Interface

This module provides the main investment agent that combines:
1. ReAct reasoning agent
2. Specialized RAG tools
3. Data review tools
4. Deep research workflow
5. Chat interface
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class InvestmentAgent:
    """
    Main investment agent that provides both deep research and chat capabilities
    
    This agent can:
    1. Run comprehensive analysis (deep research)
    2. Enable chat interface with ReAct reasoning
    3. Use specialized tools for different data types
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the investment agent
        
        Args:
            llm_client: LLM client for reasoning and response generation
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.react_agent = None
        self.rag_tool_manager = None
        self.data_tools = {}
        
        # State management
        self.chat_enabled = False
        self.current_analysis = None
        self.current_ticker = None
        
        # Initialize the system
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all agent components"""
        try:
            # Import components
            from .react_agent import ReActAgent
            from .rag_tools import RAGToolManager, CompanyNewsRAGTool, EconomicDataRAGTool, SECFilingsRAGTool
            from .data_tools import review_stock_data, review_technical_analysis, get_tool_descriptions
            
            # Initialize ReAct agent
            self.react_agent = ReActAgent(llm_client=self.llm_client)
            
            # Initialize RAG tool manager
            self.rag_tool_manager = RAGToolManager()
            
            # Register data tools
            self.data_tools = {
                'review_stock_data': review_stock_data,
                'review_technical_analysis': review_technical_analysis
            }
            
            # Get tool descriptions
            data_tool_descriptions = get_tool_descriptions()
            
            # Register all tools with ReAct agent
            all_tools = self.data_tools.copy()
            all_descriptions = data_tool_descriptions.copy()
            
            # Add RAG tools (these will be initialized when vector stores are available)
            rag_tool_descriptions = {
                'rag_company_news': 'Search for company-specific news, press releases, and recent developments',
                'rag_current_economics': 'Search for economic indicators, Fed policy, market conditions, and macroeconomic data',
                'rag_filings': 'Search for SEC filings, financial reports, earnings calls, and company financial data'
            }
            
            all_descriptions.update(rag_tool_descriptions)
            
            # Register tools with ReAct agent
            self.react_agent.register_tools(all_tools, all_descriptions)
            
            self.logger.info("Investment agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing investment agent: {e}")
            raise
    
    def run_deep_research(self, ticker: str) -> Dict[str, Any]:
        """
        Run comprehensive analysis using all tools
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Complete analysis results with recommendation
        """
        try:
            self.logger.info(f"Starting deep research for {ticker}")
            
            # Initialize RAG tools if not already done
            self._initialize_rag_tools(ticker)
            
            # Run all tools for comprehensive analysis
            results = {}
            
            # 1. Stock data
            self.logger.info("Getting stock data...")
            results['stock_data'] = self.data_tools['review_stock_data'](ticker)
            
            # 2. Technical analysis
            self.logger.info("Performing technical analysis...")
            results['technical_analysis'] = self.data_tools['review_technical_analysis'](ticker)
            
            # 3. Company news
            self.logger.info("Searching company news...")
            if 'rag_company_news' in self.react_agent.tools:
                results['company_news'] = self.react_agent.tools['rag_company_news'](
                    query="recent developments and announcements", 
                    ticker=ticker
                )
            
            # 4. Economic data
            self.logger.info("Getting economic context...")
            if 'rag_current_economics' in self.react_agent.tools:
                results['economic_data'] = self.react_agent.tools['rag_current_economics'](
                    query="current market conditions and economic indicators"
                )
            
            # 5. SEC filings
            self.logger.info("Searching SEC filings...")
            if 'rag_filings' in self.react_agent.tools:
                results['sec_filings'] = self.react_agent.tools['rag_filings'](
                    query="financial performance and recent earnings", 
                    ticker=ticker
                )
            
            # Generate recommendation using existing recommendation engine
            recommendation = self._generate_recommendation(results)
            
            # Store results and enable chat
            self.current_analysis = results
            self.current_ticker = ticker
            self.chat_enabled = True
            
            # Set context for ReAct agent
            self.react_agent.set_context({
                'ticker': ticker,
                'analysis_type': 'deep_research',
                'analysis_results': results
            })
            
            self.logger.info(f"Deep research completed for {ticker}")
            
            return {
                'ticker': ticker,
                'analysis': results,
                'recommendation': recommendation,
                'chat_enabled': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in deep research: {e}")
            return {
                'error': f"Failed to complete deep research: {str(e)}",
                'ticker': ticker,
                'chat_enabled': False
            }
    
    def chat(self, user_message: str) -> str:
        """
        Process user chat message using ReAct reasoning
        
        Args:
            user_message: User's question or request
            
        Returns:
            Agent's response
        """
        if not self.chat_enabled:
            return "Chat is not enabled. Please run deep research first to enable the chat interface."
        
        if not self.react_agent:
            return "Agent not properly initialized. Please try again."
        
        try:
            # Process chat using ReAct agent
            response = self.react_agent.process_chat(user_message)
            return response
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            return f"I encountered an error while processing your message: {str(e)}"
    
    def _initialize_rag_tools(self, ticker: str):
        """Initialize RAG tools with vector stores"""
        try:
            # Import vector store and embedding model
            from src.rag.vector_store import FAISSVectorStore
            from sentence_transformers import SentenceTransformer
            import config
            
            # Initialize embedding model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vector stores for different document types
            # In a real implementation, you'd have separate vector stores for each domain
            
            # For now, we'll create mock RAG tools
            self._create_mock_rag_tools(ticker)
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG tools: {e}")
            # Continue without RAG tools for now
    
    def _create_mock_rag_tools(self, ticker: str):
        """Create mock RAG tools for development"""
        try:
            from .rag_tools import CompanyNewsRAGTool, EconomicDataRAGTool, SECFilingsRAGTool
            
            # Create mock vector stores (in real implementation, these would be actual FAISS indexes)
            class MockVectorStore:
                def search(self, query, top_k=5):
                    # Return mock results
                    return [
                        {
                            'title': f'Mock result for {query}',
                            'content': f'This is a mock result for the query: {query}',
                            'relevance_score': 0.85,
                            'source': 'mock'
                        }
                    ]
            
            mock_vector_store = MockVectorStore()
            mock_embedding_model = None  # Not needed for mock
            
            # Create RAG tools
            company_news_tool = CompanyNewsRAGTool(mock_vector_store, mock_embedding_model)
            economic_data_tool = EconomicDataRAGTool(mock_vector_store, mock_embedding_model)
            sec_filings_tool = SECFilingsRAGTool(mock_vector_store, mock_embedding_model)
            
            # Register with ReAct agent
            rag_tools = {
                'rag_company_news': lambda **kwargs: company_news_tool.search(**kwargs),
                'rag_current_economics': lambda **kwargs: economic_data_tool.search(**kwargs),
                'rag_filings': lambda **kwargs: sec_filings_tool.search(**kwargs)
            }
            
            self.react_agent.tools.update(rag_tools)
            
            self.logger.info("Mock RAG tools created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating mock RAG tools: {e}")
    
    def _generate_recommendation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendation using existing recommendation engine"""
        try:
            # Import recommendation engine
            from src.analysis.recommendation_engine import RecommendationEngine
            
            # Extract data for recommendation engine
            stock_data = results.get('stock_data', {})
            technical_analysis = results.get('technical_analysis', {})
            
            # Create mock economic context and RAG insights for now
            economic_context = {
                'economic_indicators': {
                    'economic_sentiment': {'overall_sentiment': 'neutral'},
                    'fed_rate': {'change_direction': 'hold'},
                    'market_volatility': {'volatility_regime': 'normal'},
                    'inflation': {'trend': 'stable'}
                }
            }
            
            rag_insights = {
                'market_sentiment': 'neutral',
                'key_themes': ['general market conditions'],
                'relevant_documents': [],
                'fundamental_summary': 'Standard market conditions'
            }
            
            # Generate recommendations
            recommendation_engine = RecommendationEngine()
            recommendations = recommendation_engine.generate_recommendations(
                fundamental_data=stock_data,
                technical_analysis=technical_analysis,
                economic_context=economic_context,
                rag_insights=rag_insights
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return {
                'error': f"Failed to generate recommendation: {str(e)}",
                'short_term': {'recommendation': 'HOLD', 'confidence': 'Low'},
                'long_term': {'recommendation': 'HOLD', 'confidence': 'Low'}
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'chat_enabled': self.chat_enabled,
            'current_ticker': self.current_ticker,
            'has_analysis': self.current_analysis is not None,
            'tools_available': list(self.react_agent.tools.keys()) if self.react_agent else [],
            'llm_configured': self.llm_client is not None
        }
    
    def reset(self):
        """Reset agent state"""
        self.chat_enabled = False
        self.current_analysis = None
        self.current_ticker = None
        if self.react_agent:
            self.react_agent.conversation_history = []
            self.react_agent.current_context = {}
        self.logger.info("Agent state reset")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        if self.react_agent:
            return self.react_agent.conversation_history
        return []
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of current analysis"""
        if not self.current_analysis:
            return {'error': 'No analysis available'}
        
        summary = {
            'ticker': self.current_ticker,
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Add summaries for each component
        if 'stock_data' in self.current_analysis:
            summary['components']['stock_data'] = {
                'current_price': self.current_analysis['stock_data'].get('current_price'),
                'market_cap': self.current_analysis['stock_data'].get('market_cap'),
                'pe_ratio': self.current_analysis['stock_data'].get('pe_ratio')
            }
        
        if 'technical_analysis' in self.current_analysis:
            summary['components']['technical_analysis'] = {
                'overall_signal': self.current_analysis['technical_analysis'].get('overall_signal'),
                'confidence': self.current_analysis['technical_analysis'].get('confidence')
            }
        
        return summary 