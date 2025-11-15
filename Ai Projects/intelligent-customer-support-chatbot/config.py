"""
Configuration settings for the Intelligent Customer Support Chatbot
"""
from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Application
    app_name: str = "Intelligent Customer Support Chatbot"
    app_version: str = "1.0.0"
    debug: bool = True
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key")
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-large"
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/chatbot_db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Vector Database
    chroma_persist_dir: str = "./chroma_db"
    
    # Conversation Settings
    max_conversation_turns: int = 50
    session_timeout: int = 1800  # seconds
    max_message_length: int = 2000
    context_window: int = 10
    
    # Intent Classification
    intent_confidence_threshold: float = 0.7
    sentiment_threshold: float = -0.5
    
    # Escalation
    auto_escalate_keywords: List[str] = ["agent", "human", "representative", "speak to someone"]
    escalation_confidence_threshold: float = 0.7
    max_attempts: int = 3
    
    # LLM Settings
    temperature: float = 0.7
    max_tokens: int = 500
    timeout: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

