"""
Base agent class for all airline agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Document, Settings
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key with fallback
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")

Settings.llm = OpenAI(
    model="gpt-4",
    temperature=0.1,
    api_key=api_key if api_key else None
)

Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-ada-002",
    api_key=api_key if api_key else None
)


class BaseAgent(ABC):
    """Base class for all airline agents"""
    
    def __init__(self, name: str, description: str, db: Session):
        self.name = name
        self.description = description
        self.db = db
        self.llm = Settings.llm
        self.vector_index = None
        self._setup_knowledge_base()
    
    def _setup_knowledge_base(self):
        """Setup knowledge base for the agent"""
        knowledge_docs = self.get_knowledge_documents()
        if knowledge_docs:
            documents = [Document(text=doc) for doc in knowledge_docs]
            self.vector_index = VectorStoreIndex.from_documents(documents)
    
    def get_knowledge_documents(self) -> list[str]:
        """Override in subclasses to provide domain-specific knowledge"""
        return []
    
    def query_knowledge_base(self, query: str) -> Optional[str]:
        """Query the agent's knowledge base"""
        if not self.vector_index:
            return None
        query_engine = self.vector_index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
    
    @abstractmethod
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user input and return response"""
        pass
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using LLM"""
        full_prompt = f"""
        You are {self.name}, {self.description}
        
        Context: {context}
        
        User Query: {prompt}
        
        Provide a helpful, accurate response:
        """
        
        response = self.llm.complete(full_prompt)
        return str(response)


