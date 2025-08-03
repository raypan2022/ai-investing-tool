"""
ReAct Agent for Investment Analysis

This module implements a ReAct (Reasoning and Acting) agent that can:
1. Analyze user intent
2. Select appropriate tools
3. Execute tools and gather results
4. Generate comprehensive responses
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct agent for investment analysis
    
    This agent uses reasoning to determine which tools to use and
    generates comprehensive responses based on tool results.
    """
    
    def __init__(self, llm_client=None, max_iterations: int = 5):
        """
        Initialize the ReAct agent
        
        Args:
            llm_client: LLM client for reasoning and response generation
            max_iterations: Maximum number of reasoning iterations
        """
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.logger = logging.getLogger(__name__)
        
        # Tool registry
        self.tools = {}
        self.tool_descriptions = {}
        
        # Context management
        self.conversation_history = []
        self.current_context = {}
    
    def register_tools(self, tools: Dict[str, Any], descriptions: Dict[str, str]):
        """Register tools and their descriptions"""
        self.tools.update(tools)
        self.tool_descriptions.update(descriptions)
        self.logger.info(f"Registered {len(tools)} tools")
    
    def set_context(self, context: Dict[str, Any]):
        """Set conversation context (e.g., current ticker, previous analysis)"""
        self.current_context = context
        self.logger.info("Context updated")
    
    def add_to_history(self, user_message: str, agent_response: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            'user': user_message,
            'agent': agent_response,
            'timestamp': datetime.now().isoformat()
        })
    
    def process_chat(self, user_message: str) -> str:
        """
        Process user chat message using ReAct reasoning
        
        Args:
            user_message: User's question or request
            
        Returns:
            Agent's response
        """
        try:
            self.logger.info(f"Processing chat: {user_message}")
            
            # Initialize reasoning loop
            thoughts = []
            observations = []
            actions_taken = []
            
            for iteration in range(self.max_iterations):
                # Step 1: Think - Analyze what we need to do
                thought = self._think(user_message, thoughts, observations, actions_taken)
                thoughts.append(thought)
                
                # Step 2: Act - Decide which tool to use
                action = self._act(thought, user_message)
                if not action:
                    break
                
                actions_taken.append(action)
                
                # Step 3: Observe - Execute tool and get results
                observation = self._observe(action)
                observations.append(observation)
                
                # Check if we have enough information
                if self._should_stop(thought, observation):
                    break
            
            # Step 4: Respond - Generate final answer
            response = self._respond(user_message, thoughts, observations, actions_taken)
            
            # Add to history
            self.add_to_history(user_message, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing chat: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _think(self, user_message: str, thoughts: List[str], observations: List[str], actions: List[str]) -> str:
        """
        Think step - Analyze what needs to be done
        
        Args:
            user_message: Original user message
            thoughts: Previous thoughts
            observations: Previous observations
            actions: Previous actions taken
            
        Returns:
            Current thought about what to do next
        """
        if not self.llm_client:
            # Fallback reasoning without LLM
            return self._fallback_reasoning(user_message, thoughts, observations, actions)
        
        # Build context for reasoning
        context = self._build_reasoning_context(user_message, thoughts, observations, actions)
        
        # Generate reasoning prompt
        prompt = f"""You are an investment analysis agent. Analyze the user's request and determine what information you need.

User Request: {user_message}

Available Tools:
{self._format_tool_descriptions()}

Previous Thoughts: {thoughts[-3:] if thoughts else 'None'}
Previous Observations: {observations[-3:] if observations else 'None'}
Previous Actions: {actions[-3:] if actions else 'None'}

Current Context: {self.current_context}

Think step by step about what the user is asking for and which tools would be most helpful.

Thought:"""
        
        # Get reasoning from LLM
        try:
            response = self._call_llm(prompt, max_tokens=200)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error in thinking step: {e}")
            return self._fallback_reasoning(user_message, thoughts, observations, actions)
    
    def _act(self, thought: str, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Act step - Decide which tool to use
        
        Args:
            thought: Current reasoning
            user_message: Original user message
            
        Returns:
            Action to take (tool name and parameters)
        """
        if not self.llm_client:
            return self._fallback_action(thought, user_message)
        
        # Generate action prompt
        prompt = f"""Based on your reasoning, decide which tool to use.

Thought: {thought}
User Request: {user_message}

Available Tools:
{self._format_tool_descriptions()}

Respond with a JSON object in this format:
{{
    "tool": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "Why this tool is appropriate"
}}

If no tool is needed, respond with:
{{
    "tool": null,
    "reasoning": "No tool needed, ready to respond"
}}

Action:"""
        
        try:
            response = self._call_llm(prompt, max_tokens=150)
            action_data = json.loads(response)
            
            if action_data.get('tool'):
                return {
                    'tool': action_data['tool'],
                    'parameters': action_data.get('parameters', {}),
                    'reasoning': action_data.get('reasoning', '')
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error in action step: {e}")
            return self._fallback_action(thought, user_message)
    
    def _observe(self, action: Dict[str, Any]) -> str:
        """
        Observe step - Execute tool and get results
        
        Args:
            action: Action to execute
            
        Returns:
            Observation (tool results)
        """
        tool_name = action['tool']
        parameters = action.get('parameters', {})
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            # Execute the tool
            tool = self.tools[tool_name]
            result = tool(**parameters)
            
            # Format the result
            if hasattr(tool, 'format_results'):
                formatted_result = tool.format_results(result)
            else:
                formatted_result = str(result)
            
            self.logger.info(f"Executed tool {tool_name} successfully")
            return f"Tool {tool_name} results:\n{formatted_result}"
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing tool {tool_name}: {str(e)}"
    
    def _respond(self, user_message: str, thoughts: List[str], observations: List[str], actions: List[str]) -> str:
        """
        Respond step - Generate final answer
        
        Args:
            user_message: Original user message
            thoughts: All thoughts from reasoning
            observations: All observations from tools
            actions: All actions taken
            
        Returns:
            Final response to user
        """
        if not self.llm_client:
            return self._fallback_response(user_message, thoughts, observations, actions)
        
        # Build response prompt
        prompt = f"""You are an investment analysis assistant. Generate a comprehensive response to the user's question.

User Question: {user_message}

Reasoning Process:
{self._format_reasoning_process(thoughts, observations, actions)}

Current Context: {self.current_context}

Generate a helpful, accurate, and comprehensive response based on the information gathered. 
Be professional, clear, and actionable in your response.

Response:"""
        
        try:
            response = self._call_llm(prompt, max_tokens=500)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._fallback_response(user_message, thoughts, observations, actions)
    
    def _should_stop(self, thought: str, observation: str) -> bool:
        """Determine if we should stop the reasoning loop"""
        # Stop if we have enough information or hit an error
        if "error" in observation.lower():
            return True
        
        # Stop if the thought indicates we're ready to respond
        if any(keyword in thought.lower() for keyword in ['ready to respond', 'have enough information', 'can answer now']):
            return True
        
        return False
    
    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Call the LLM client"""
        if not self.llm_client:
            raise ValueError("No LLM client configured")
        
        # This is where you'd integrate with your actual LLM
        # For now, return a placeholder
        return f"LLM response would be generated here for prompt: {prompt[:100]}..."
    
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for prompts"""
        descriptions = []
        for tool_name, description in self.tool_descriptions.items():
            descriptions.append(f"- {tool_name}: {description}")
        return "\n".join(descriptions)
    
    def _format_reasoning_process(self, thoughts: List[str], observations: List[str], actions: List[str]) -> str:
        """Format the reasoning process for response generation"""
        process = []
        for i, (thought, action, observation) in enumerate(zip(thoughts, actions, observations), 1):
            process.append(f"Step {i}:")
            process.append(f"  Thought: {thought}")
            if action:
                process.append(f"  Action: {action['tool']} - {action.get('reasoning', '')}")
            process.append(f"  Observation: {observation}")
            process.append("")
        return "\n".join(process)
    
    def _build_reasoning_context(self, user_message: str, thoughts: List[str], observations: List[str], actions: List[str]) -> str:
        """Build context for reasoning"""
        context_parts = []
        
        if self.current_context:
            context_parts.append(f"Current ticker: {self.current_context.get('ticker', 'None')}")
            context_parts.append(f"Previous analysis: {self.current_context.get('analysis_type', 'None')}")
        
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            context_parts.append("Recent conversation:")
            for exchange in recent_history:
                context_parts.append(f"  User: {exchange['user']}")
                context_parts.append(f"  Agent: {exchange['agent'][:100]}...")
        
        return "\n".join(context_parts) if context_parts else "No additional context"
    
    # Fallback methods for when LLM is not available
    def _fallback_reasoning(self, user_message: str, thoughts: List[str], observations: List[str], actions: List[str]) -> str:
        """Fallback reasoning without LLM"""
        # Simple keyword-based reasoning
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ['price', 'stock', 'market cap', 'pe ratio']):
            return "User is asking about stock data and fundamentals. I should use review_stock_data tool."
        elif any(word in user_lower for word in ['technical', 'rsi', 'macd', 'indicator']):
            return "User is asking about technical analysis. I should use review_technical_analysis tool."
        elif any(word in user_lower for word in ['news', 'announcement', 'press release']):
            return "User is asking about company news. I should use rag_company_news tool."
        elif any(word in user_lower for word in ['economic', 'fed', 'inflation', 'market condition']):
            return "User is asking about economic conditions. I should use rag_current_economics tool."
        elif any(word in user_lower for word in ['earnings', 'filing', 'financial', 'sec']):
            return "User is asking about financial data. I should use rag_filings tool."
        else:
            return "User's request is unclear. I should gather more information using available tools."
    
    def _fallback_action(self, thought: str, user_message: str) -> Optional[Dict[str, Any]]:
        """Fallback action selection without LLM"""
        user_lower = user_message.lower()
        ticker = self.current_context.get('ticker', '')
        
        if 'stock data' in thought.lower():
            return {'tool': 'review_stock_data', 'parameters': {'ticker': ticker}, 'reasoning': 'Need stock fundamentals'}
        elif 'technical' in thought.lower():
            return {'tool': 'review_technical_analysis', 'parameters': {'ticker': ticker}, 'reasoning': 'Need technical indicators'}
        elif 'news' in thought.lower():
            return {'tool': 'rag_company_news', 'parameters': {'ticker': ticker, 'query': user_message}, 'reasoning': 'Need company news'}
        elif 'economic' in thought.lower():
            return {'tool': 'rag_current_economics', 'parameters': {'query': user_message}, 'reasoning': 'Need economic data'}
        elif 'financial' in thought.lower():
            return {'tool': 'rag_filings', 'parameters': {'ticker': ticker, 'query': user_message}, 'reasoning': 'Need financial data'}
        
        return None
    
    def _fallback_response(self, user_message: str, thoughts: List[str], observations: List[str], actions: List[str]) -> str:
        """Fallback response generation without LLM"""
        if not observations:
            return "I don't have enough information to answer your question. Please try asking about specific stock data, technical indicators, news, or economic conditions."
        
        # Combine observations into a simple response
        response_parts = [f"Based on the information I gathered:"]
        
        for i, observation in enumerate(observations, 1):
            if "error" not in observation.lower():
                response_parts.append(f"\n{observation}")
        
        if len(response_parts) == 1:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or asking about a different aspect."
        
        return "\n".join(response_parts) 