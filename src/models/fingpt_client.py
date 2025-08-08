"""FinGPT client for self-hosted model on Modal or similar platform"""

import requests
from typing import List, Dict, Any
import os


class FinGPTClient:
    """Client for self-hosted FinGPT model"""
    
    def __init__(self, 
                 endpoint_url: str = None,
                 api_key: str = None,
                 model_name: str = "fingpt-7b"):
        """
        Initialize FinGPT client for self-hosted model
        
        Args:
            endpoint_url: URL of your hosted FinGPT endpoint (Modal, RunPod, etc.)
            api_key: API key for authentication
            model_name: Name of the FinGPT model variant
        """
        self.endpoint_url = endpoint_url or os.getenv("FINGPT_ENDPOINT_URL")
        self.api_key = api_key or os.getenv("FINGPT_API_KEY")
        self.model_name = model_name
        
        if not self.endpoint_url:
            raise ValueError("FinGPT endpoint URL not provided")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.1,
                       max_tokens: int = 1000) -> str:
        """
        Send chat completion request to hosted FinGPT model
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        # Implementation will depend on your hosting setup
        # This is a generic structure for Modal/API-based hosting
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": self.model_name
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                f"{self.endpoint_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"Error calling FinGPT: {str(e)}"
    
    def bind_tools(self, tools: List[Any]):
        """
        Bind tools to the model (similar to LangChain's bind_tools)
        For FinGPT, we'll need to format tool descriptions in prompts
        """
        self.tools = tools
        self.tool_descriptions = self._format_tool_descriptions(tools)
        return self
    
    def _format_tool_descriptions(self, tools: List[Any]) -> str:
        """Format tool descriptions for FinGPT prompt"""
        descriptions = []
        for tool in tools:
            name = tool.__name__
            doc = tool.__doc__ or "No description available"
            descriptions.append(f"- {name}: {doc.strip()}")
        
        return "\n".join(descriptions)
    
    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Invoke the model with tool-aware system prompt
        This method mimics LangChain's interface for compatibility
        """
        # Add system prompt with tool information
        system_prompt = self._create_system_prompt()
        
        # Prepare messages with system prompt
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Get response from FinGPT
        response_content = self.chat_completion(full_messages)
        
        # Parse for tool calls (simplified - you'll need to implement tool call parsing)
        tool_calls = self._parse_tool_calls(response_content)
        
        # Return in LangChain-compatible format
        return type('AIMessage', (), {
            'content': response_content,
            'tool_calls': tool_calls
        })()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with tool information"""
        if not hasattr(self, 'tools'):
            return "You are FinGPT, a financial analysis AI assistant."
        
        return f"""You are FinGPT, a specialized financial analysis AI assistant.

You have access to the following tools:
{self.tool_descriptions}

When you need to use a tool, format your response like this:
Tool: tool_name
Args: {{"param1": "value1", "param2": "value2"}}

After using tools, provide your analysis and recommendations based on the data."""
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from FinGPT response
        This is a simplified implementation - you'll need to make this more robust
        """
        tool_calls = []
        lines = response.split('\n')
        
        for i, line in enumerate(lines):
            if line.startswith('Tool:'):
                tool_name = line.replace('Tool:', '').strip()
                
                # Look for Args on next line
                if i + 1 < len(lines) and lines[i + 1].startswith('Args:'):
                    args_str = lines[i + 1].replace('Args:', '').strip()
                    try:
                        import json
                        args = json.loads(args_str)
                        tool_calls.append({
                            'function': {
                                'name': tool_name,
                                'arguments': json.dumps(args)
                            }
                        })
                    except json.JSONDecodeError:
                        continue
        
        return tool_calls