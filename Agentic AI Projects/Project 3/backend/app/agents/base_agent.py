"""
Base agent class for all AI agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.utils.gemini_client import gemini_client


class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.gemini = gemini_client
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's main task"""
        pass
    
    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Generate response using Gemini"""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        return self.gemini.generate_text(
            full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=self.system_prompt
        )
    
    def generate_structured(
        self,
        prompt: str,
        format_description: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate structured output"""
        return self.gemini.generate_structured_output(
            prompt,
            format_description,
            temperature
        )

