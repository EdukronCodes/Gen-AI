"""
Configuration settings for Autonomous Business Intelligence System
"""
from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Application
    app_name: str = "Autonomous Business Intelligence"
    app_version: str = "1.0.0"
    debug: bool = True
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key")
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8001
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4-turbo-preview"
    
    # Databases
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/bi_db")
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017/bi_analytics")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Agent Configuration
    max_concurrent_tasks: int = 100
    agent_timeout: int = 300
    retry_attempts: int = 3
    learning_enabled: bool = True
    
    # Data Processing
    batch_size: int = 10000
    parallel_connections: int = 10
    refresh_interval: int = 3600
    
    # Analysis Settings
    max_analysis_depth: int = 5
    statistical_significance: float = 0.05
    correlation_threshold: float = 0.7
    
    # Anomaly Detection
    anomaly_sensitivity: float = 0.8
    window_size: int = 30
    alert_threshold: float = 3.0
    
    # Predictive Analytics
    forecast_horizon: int = 90
    confidence_interval: float = 0.95
    model_retrain_frequency: str = "weekly"
    
    # Insights
    insight_generation_frequency: str = "hourly"
    min_confidence: float = 0.7
    max_insights_per_category: int = 10
    retention_days: int = 90
    
    # LLM Settings
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

