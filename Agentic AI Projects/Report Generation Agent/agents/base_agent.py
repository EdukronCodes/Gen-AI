"""
Base Agent Class for Multi-Agent Framework
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class BaseAgent(ABC):
    """Base class for all agents in the framework"""
    
    def __init__(self, name: str, role: str, gemini_client=None):
        self.name = name
        self.role = role
        # Use provided client or create Gemini client
        if gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        else:
            self.gemini_model = gemini_client
        self.tools = []
        self.conversation_history = []
    
    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the agent's primary task
        Args:
            task: The task description
            context: Additional context from other agents
        Returns:
            Dictionary with results and metadata
        """
        pass
    
    def add_tool(self, tool):
        """Add a tool to the agent's toolkit"""
        self.tools.append(tool)
    
    async def _call_gemini(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """Helper method to call Gemini API"""
        # Combine system prompt and user prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Generate content using Gemini (simplified like the working example)
        response = self.gemini_model.generate_content(full_prompt)
        
        return response.text
    
    def log(self, message: str):
        """Log agent activity"""
        print(f"[{self.name}] {message}")

