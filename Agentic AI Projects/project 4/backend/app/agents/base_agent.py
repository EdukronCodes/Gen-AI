"""
Base Agent Class
All agents inherit from this base class
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from app.core.config import settings
import json


class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, name: str, role: str, goal: str):
        self.name = name
        self.role = role
        self.goal = goal
        
        # Initialize LLM
        if settings.USE_AZURE:
            from langchain_openai import AzureChatOpenAI
            self.llm = AzureChatOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
                temperature=0.7
            )
        else:
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                api_key=settings.OPENAI_API_KEY
            )
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return result"""
        pass
    
    def log_action(self, action: str, data: Dict[str, Any]):
        """Log agent action"""
        print(f"[{self.name}] {action}: {json.dumps(data, indent=2)}")


