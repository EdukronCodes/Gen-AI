"""
Configuration file for Agentic RAG-Based Medical Chatbot
Contains all API keys, paths, and system configurations
"""

import os
from typing import Optional

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key-here")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Medical Knowledge Base Paths
MEDICAL_KNOWLEDGE_BASE_PATH = os.getenv("MEDICAL_KNOWLEDGE_BASE_PATH", "./medical_knowledge")
PUBMED_DATA_PATH = os.getenv("PUBMED_DATA_PATH", "./data/pubmed")
CLINICAL_GUIDELINES_PATH = os.getenv("CLINICAL_GUIDELINES_PATH", "./data/guidelines")
DRUG_DATABASE_PATH = os.getenv("DRUG_DATABASE_PATH", "./data/drugs")

# Model Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# RAG Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")  # chroma or pinecone
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Agent Configuration
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "30"))
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_THRESHOLD", "0.8"))

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_chatbot.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/medical_chatbot.log")

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Medical Specific Configuration
EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "severe bleeding",
    "unconscious", "stroke", "heart attack", "seizure"
]

URGENCY_LEVELS = {
    1: "Low - General information request",
    2: "Mild - Minor symptoms or concerns",
    3: "Moderate - Concerning symptoms",
    4: "High - Serious symptoms requiring attention",
    5: "Critical - Emergency situation"
}

QUERY_TYPES = {
    "symptom": "Symptom analysis and assessment",
    "medication": "Medication information and interactions",
    "treatment": "Treatment recommendations",
    "emergency": "Emergency assessment and triage"
}

# Knowledge Base Sources
KNOWLEDGE_SOURCES = {
    "pubmed_articles": {
        "path": PUBMED_DATA_PATH,
        "type": "research_articles",
        "update_frequency": "weekly"
    },
    "clinical_guidelines": {
        "path": CLINICAL_GUIDELINES_PATH,
        "type": "guidelines",
        "update_frequency": "monthly"
    },
    "drug_database": {
        "path": DRUG_DATABASE_PATH,
        "type": "drug_info",
        "update_frequency": "daily"
    },
    "medical_textbooks": {
        "path": f"{MEDICAL_KNOWLEDGE_BASE_PATH}/textbooks",
        "type": "reference_material",
        "update_frequency": "quarterly"
    },
    "symptom_checker": {
        "path": f"{MEDICAL_KNOWLEDGE_BASE_PATH}/symptoms",
        "type": "symptom_database",
        "update_frequency": "monthly"
    }
}

# Agent Specific Configurations
AGENT_CONFIGS = {
    "symptom_analysis": {
        "model": "medical-bert-symptom-classifier",
        "confidence_threshold": 0.7,
        "max_symptoms_per_query": 5
    },
    "medication_info": {
        "model": "drug-interaction-checker",
        "confidence_threshold": 0.8,
        "max_medications_per_query": 10
    },
    "treatment_recommendation": {
        "model": "treatment-recommendation-engine",
        "confidence_threshold": 0.75,
        "max_recommendations": 3
    },
    "emergency_assessment": {
        "model": "emergency-triage-system",
        "confidence_threshold": 0.9,
        "response_time_limit": 5
    }
}

# Validation Functions
def validate_config():
    """Validate configuration settings"""
    required_vars = [
        "OPENAI_API_KEY",
        "SECRET_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not globals().get(var) or globals().get(var).startswith("your-"):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required configuration variables: {missing_vars}")
    
    # Validate paths
    paths_to_create = [
        CHROMA_DB_PATH,
        MEDICAL_KNOWLEDGE_BASE_PATH,
        os.path.dirname(LOG_FILE)
    ]
    
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)

# Initialize configuration
if __name__ == "__main__":
    validate_config()
    print("Configuration validated successfully!") 