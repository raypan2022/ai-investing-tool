"""
Specialized RAG Tools for Investment Analysis

This module provides specialized RAG tools for different document types:
- Company news and press releases
- Economic indicators and market data
- SEC filings and financial documents

Each tool is optimized for its specific domain and document type.
"""

from typing import Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RAGToolBase(ABC):
    """Base class for all RAG tools"""
    
    def __init__(self, vector_store, embedding_model, tool_name: str):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.tool_name = tool_name
        self.logger = logging.getLogger(f"{__name__}.{tool_name}")
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Search and return relevant documents"""
        pass
    
    @abstractmethod
    def get_tool_description(self) -> str:
        """Get description of what this tool does"""
        pass
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM consumption"""
        if not results:
            return "No relevant documents found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result.get('title', 'Untitled')}")
            formatted.append(f"   Content: {result.get('content', '')[:200]}...")
            formatted.append(f"   Relevance: {result.get('relevance_score', 0):.2f}")
            formatted.append("")
        
        return "\n".join(formatted)


class CompanyNewsRAGTool(RAGToolBase):
    """Specialized RAG tool for company news and press releases"""
    
    def __init__(self, vector_store, embedding_model):
        super().__init__(vector_store, embedding_model, "CompanyNewsRAG")
    
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search company news and press releases
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional parameters (ticker, date_range, etc.)
        """
        try:
            ticker = kwargs.get('ticker', '')
            
            # Enhance query with company context
            enhanced_query = f"{ticker} {query}" if ticker else query
            
            # Search vector store
            results = self.vector_store.search(enhanced_query, top_k=top_k)
            
            # Filter and rank by relevance and recency
            filtered_results = []
            for result in results:
                # Add company news specific metadata
                result['source_type'] = 'company_news'
                result['tool_used'] = self.tool_name
                
                # Filter by relevance threshold
                if result.get('relevance_score', 0) > 0.7:
                    filtered_results.append(result)
            
            self.logger.info(f"Found {len(filtered_results)} relevant news articles for query: {query}")
            return filtered_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error searching company news: {e}")
            return []
    
    def get_tool_description(self) -> str:
        return """Search for company-specific news, press releases, and recent developments. 
        Use this tool when users ask about recent company news, announcements, or developments."""


class EconomicDataRAGTool(RAGToolBase):
    """Specialized RAG tool for economic indicators and market data"""
    
    def __init__(self, vector_store, embedding_model):
        super().__init__(vector_store, embedding_model, "EconomicDataRAG")
    
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search economic indicators and market conditions
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional parameters (date_range, indicator_type, etc.)
        """
        try:
            # Enhance query with economic context
            enhanced_query = f"economic indicators market data {query}"
            
            # Search vector store
            results = self.vector_store.search(enhanced_query, top_k=top_k)
            
            # Filter and rank by relevance
            filtered_results = []
            for result in results:
                # Add economic data specific metadata
                result['source_type'] = 'economic_data'
                result['tool_used'] = self.tool_name
                
                # Filter by relevance threshold
                if result.get('relevance_score', 0) > 0.6:
                    filtered_results.append(result)
            
            self.logger.info(f"Found {len(filtered_results)} relevant economic data points for query: {query}")
            return filtered_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error searching economic data: {e}")
            return []
    
    def get_tool_description(self) -> str:
        return """Search for economic indicators, Fed policy, market conditions, and macroeconomic data. 
        Use this tool when users ask about economic environment, market conditions, or macroeconomic factors."""


class SECFilingsRAGTool(RAGToolBase):
    """Specialized RAG tool for SEC filings and financial documents"""
    
    def __init__(self, vector_store, embedding_model):
        super().__init__(vector_store, embedding_model, "SECFilingsRAG")
    
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search SEC filings and financial documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional parameters (ticker, filing_type, date_range, etc.)
        """
        try:
            ticker = kwargs.get('ticker', '')
            filing_type = kwargs.get('filing_type', '')
            
            # Enhance query with financial context
            enhanced_query = f"{ticker} SEC filing financial {query}" if ticker else f"SEC filing financial {query}"
            
            # Search vector store
            results = self.vector_store.search(enhanced_query, top_k=top_k)
            
            # Filter and rank by relevance and filing type
            filtered_results = []
            for result in results:
                # Add SEC filing specific metadata
                result['source_type'] = 'sec_filing'
                result['tool_used'] = self.tool_name
                
                # Filter by relevance threshold
                if result.get('relevance_score', 0) > 0.7:
                    filtered_results.append(result)
            
            self.logger.info(f"Found {len(filtered_results)} relevant SEC filing sections for query: {query}")
            return filtered_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error searching SEC filings: {e}")
            return []
    
    def get_tool_description(self) -> str:
        return """Search for SEC filings, financial reports, earnings calls, and company financial data. 
        Use this tool when users ask about financial performance, earnings, or company fundamentals."""


class RAGToolManager:
    """Manager for all RAG tools"""
    
    def __init__(self):
        self.tools = {}
        self.logger = logging.getLogger(__name__)
    
    def register_tool(self, tool_name: str, tool: RAGToolBase):
        """Register a RAG tool"""
        self.tools[tool_name] = tool
        self.logger.info(f"Registered RAG tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[RAGToolBase]:
        """Get a specific RAG tool"""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, RAGToolBase]:
        """Get all registered tools"""
        return self.tools.copy()
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all tools for ReAct agent"""
        return {name: tool.get_tool_description() for name, tool in self.tools.items()}
    
    def search_with_tool(self, tool_name: str, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search using a specific tool"""
        tool = self.get_tool(tool_name)
        if tool:
            return tool.search(query, **kwargs)
        else:
            self.logger.error(f"Tool not found: {tool_name}")
            return [] 