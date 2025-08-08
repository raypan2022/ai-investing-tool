import os
from typing import Dict, Any, List
from openai import OpenAI

class OpenAIClient:
    """Simple OpenAI client for ReAct agent"""
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Send chat completion request"""
        # Implementation here
        pass
    
    def analyze_with_context(self, ticker: str, context: str, question: str) -> str:
        """Analyze stock with given context"""
        # Implementation here
        pass