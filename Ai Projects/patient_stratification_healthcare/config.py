"""
Configuration file for Patient Stratification for Personalized Healthcare Interventions
Contains all API keys, paths, and system configurations
"""

import os
import json
from typing import Dict, Any, List
from pathlib import Path

# API Keys and External Services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key-here")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./patient_stratification_chroma_db")

# Database Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "patient_stratification_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Data Paths
PATIENT_DATA_PATH = os.getenv("PATIENT_DATA_PATH", "./data/patient_data")
CLINICAL_GUIDELINES_PATH = os.getenv("CLINICAL_GUIDELINES_PATH", "./data/clinical_guidelines")
TREATMENT_PROTOCOLS_PATH = os.getenv("TREATMENT_PROTOCOLS_PATH", "./data/treatment_protocols")
RESEARCH_LITERATURE_PATH = os.getenv("RESEARCH_LITERATURE_PATH", "./data/research_literature")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "./models/patient_stratification")
RESULTS_PATH = os.getenv("RESULTS_PATH", "./results/patient_stratification")

# LLM Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# RAG Configuration
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))

# Clustering Configuration
NUM_CLUSTERS = int(os.getenv("NUM_CLUSTERS", "5"))
CLUSTERING_METHODS = ["kmeans", "dbscan", "hierarchical"]
DEFAULT_CLUSTERING_METHOD = "kmeans"

# K-Means Parameters
KMEANS_N_INIT = int(os.getenv("KMEANS_N_INIT", "10"))
KMEANS_MAX_ITER = int(os.getenv("KMEANS_MAX_ITER", "300"))
KMEANS_RANDOM_STATE = int(os.getenv("KMEANS_RANDOM_STATE", "42"))

# DBSCAN Parameters
DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", "0.5"))
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "5"))

# Hierarchical Clustering Parameters
HIERARCHICAL_LINKAGE = os.getenv("HIERARCHICAL_LINKAGE", "ward")
HIERARCHICAL_AFFINITY = os.getenv("HIERARCHICAL_AFFINITY", "euclidean")

# Feature Engineering
FEATURE_SCALING_METHOD = os.getenv("FEATURE_SCALING_METHOD", "standard")  # standard, minmax, robust
USE_PCA = os.getenv("USE_PCA", "false").lower() == "true"
PCA_N_COMPONENTS = int(os.getenv("PCA_N_COMPONENTS", "10"))

# Patient Features Configuration
PATIENT_FEATURES = {
    "numerical_features": [
        "age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic",
        "heart_rate", "cholesterol_total", "cholesterol_hdl", "cholesterol_ldl",
        "triglycerides", "blood_sugar"
    ],
    "categorical_features": [
        "gender", "smoking_status", "diabetes_status"
    ],
    "text_features": [
        "family_history", "medications", "symptoms", "diagnosis_history"
    ],
    "dict_features": [
        "lab_results", "vital_signs"
    ]
}

# Risk Assessment Configuration
RISK_THRESHOLDS = {
    "age": {
        "low": 0,
        "medium": 50,
        "high": 65
    },
    "bmi": {
        "low": 0,
        "medium": 25,
        "high": 30
    },
    "blood_pressure_systolic": {
        "low": 0,
        "medium": 120,
        "high": 140
    },
    "blood_pressure_diastolic": {
        "low": 0,
        "medium": 80,
        "high": 90
    },
    "cholesterol_total": {
        "low": 0,
        "medium": 200,
        "high": 240
    },
    "blood_sugar": {
        "low": 0,
        "medium": 100,
        "high": 126
    }
}

RISK_WEIGHTS = {
    "age": 1.0,
    "bmi": 1.5,
    "blood_pressure": 2.0,
    "cholesterol": 1.5,
    "blood_sugar": 2.0,
    "smoking": 2.0,
    "diabetes": 2.0
}

# Deep Learning Model Configuration
DEEP_LEARNING_CONFIG = {
    "input_dim": 13,
    "hidden_dims": [64, 32, 16],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10
}

# Random Forest Configuration
RANDOM_FOREST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42
}

# Monitoring and Follow-up Configuration
MONITORING_PLANS = {
    "high_risk": {
        "blood_pressure_monitoring": "Daily",
        "blood_sugar_monitoring": "Daily",
        "weight_monitoring": "Weekly",
        "cholesterol_testing": "Monthly",
        "doctor_visits": "Monthly",
        "specialist_consultation": "Quarterly"
    },
    "medium_risk": {
        "blood_pressure_monitoring": "Weekly",
        "blood_sugar_monitoring": "Weekly",
        "weight_monitoring": "Monthly",
        "cholesterol_testing": "Quarterly",
        "doctor_visits": "Quarterly",
        "specialist_consultation": "As needed"
    },
    "low_risk": {
        "blood_pressure_monitoring": "Monthly",
        "blood_sugar_monitoring": "Monthly",
        "weight_monitoring": "Quarterly",
        "cholesterol_testing": "Annually",
        "doctor_visits": "Annually",
        "specialist_consultation": "As needed"
    }
}

FOLLOW_UP_SCHEDULES = {
    "high_risk": {
        "next_appointment": "1 week",
        "blood_work": "2 weeks",
        "specialist_review": "1 month",
        "comprehensive_assessment": "3 months"
    },
    "medium_risk": {
        "next_appointment": "1 month",
        "blood_work": "3 months",
        "specialist_review": "6 months",
        "comprehensive_assessment": "1 year"
    },
    "low_risk": {
        "next_appointment": "3 months",
        "blood_work": "6 months",
        "specialist_review": "1 year",
        "comprehensive_assessment": "1 year"
    }
}

# Treatment Recommendation Configuration
TREATMENT_CATEGORIES = [
    "lifestyle_modifications",
    "medication_management",
    "dietary_recommendations",
    "exercise_guidelines",
    "monitoring_instructions",
    "preventive_measures"
]

# Knowledge Sources Configuration
KNOWLEDGE_SOURCES = {
    "clinical_guidelines": {
        "path": CLINICAL_GUIDELINES_PATH,
        "file_types": ["*.txt", "*.pdf", "*.docx"],
        "priority": "high"
    },
    "treatment_protocols": {
        "path": TREATMENT_PROTOCOLS_PATH,
        "file_types": ["*.txt", "*.pdf", "*.docx"],
        "priority": "high"
    },
    "research_literature": {
        "path": RESEARCH_LITERATURE_PATH,
        "file_types": ["*.txt", "*.pdf"],
        "priority": "medium"
    },
    "drug_database": {
        "path": "./data/drug_database",
        "file_types": ["*.json", "*.csv"],
        "priority": "medium"
    }
}

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8003"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.getenv("LOG_FILE", "./logs/patient_stratification.log")

# Security Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))

# Performance Configuration
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# Model Evaluation Configuration
EVALUATION_METRICS = [
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "adjusted_rand_score",
    "normalized_mutual_info_score"
]

CROSS_VALIDATION_FOLDS = int(os.getenv("CROSS_VALIDATION_FOLDS", "5"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

# Data Validation Configuration
DATA_VALIDATION_RULES = {
    "age": {"min": 0, "max": 120, "required": True},
    "bmi": {"min": 10, "max": 100, "required": True},
    "blood_pressure_systolic": {"min": 70, "max": 300, "required": True},
    "blood_pressure_diastolic": {"min": 40, "max": 200, "required": True},
    "heart_rate": {"min": 40, "max": 200, "required": True},
    "cholesterol_total": {"min": 50, "max": 500, "required": True},
    "cholesterol_hdl": {"min": 20, "max": 100, "required": True},
    "cholesterol_ldl": {"min": 20, "max": 300, "required": True},
    "triglycerides": {"min": 50, "max": 1000, "required": True},
    "blood_sugar": {"min": 50, "max": 500, "required": True}
}

# Alert Configuration
ALERT_THRESHOLDS = {
    "high_risk_patients_percentage": 0.3,
    "cluster_imbalance_threshold": 0.1,
    "model_confidence_threshold": 0.7
}

# Notification Configuration
NOTIFICATION_CHANNELS = {
    "email": {
        "enabled": os.getenv("EMAIL_NOTIFICATIONS", "false").lower() == "true",
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "sender_email": os.getenv("SENDER_EMAIL", ""),
        "sender_password": os.getenv("SENDER_PASSWORD", "")
    },
    "slack": {
        "enabled": os.getenv("SLACK_NOTIFICATIONS", "false").lower() == "true",
        "webhook_url": os.getenv("SLACK_WEBHOOK_URL", "")
    }
}

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        PATIENT_DATA_PATH,
        CLINICAL_GUIDELINES_PATH,
        TREATMENT_PROTOCOLS_PATH,
        RESEARCH_LITERATURE_PATH,
        MODEL_SAVE_PATH,
        RESULTS_PATH,
        os.path.dirname(LOG_FILE),
        CHROMA_DB_PATH
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_database_url() -> str:
    """Get database connection URL"""
    return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

def get_redis_url() -> str:
    """Get Redis connection URL"""
    return f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check required environment variables
        required_vars = ["OPENAI_API_KEY"]
        for var in required_vars:
            if not os.getenv(var):
                print(f"Warning: {var} not set")
        
        # Validate paths
        create_directories()
        
        # Validate numerical values
        assert NUM_CLUSTERS > 0, "NUM_CLUSTERS must be positive"
        assert API_PORT > 0, "API_PORT must be positive"
        assert TEST_SIZE > 0 and TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
        
        # Validate clustering method
        assert DEFAULT_CLUSTERING_METHOD in CLUSTERING_METHODS, f"Invalid clustering method: {DEFAULT_CLUSTERING_METHOD}"
        
        print("Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "openai_api_key": OPENAI_API_KEY,
        "pinecone_api_key": PINECONE_API_KEY,
        "chroma_db_path": CHROMA_DB_PATH,
        "postgres_config": {
            "host": POSTGRES_HOST,
            "port": POSTGRES_PORT,
            "database": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD
        },
        "redis_config": {
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "db": REDIS_DB
        },
        "data_paths": {
            "patient_data": PATIENT_DATA_PATH,
            "clinical_guidelines": CLINICAL_GUIDELINES_PATH,
            "treatment_protocols": TREATMENT_PROTOCOLS_PATH,
            "research_literature": RESEARCH_LITERATURE_PATH,
            "model_save": MODEL_SAVE_PATH,
            "results": RESULTS_PATH
        },
        "llm_config": {
            "model_name": LLM_MODEL_NAME,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS
        },
        "rag_config": {
            "chunk_size": RAG_CHUNK_SIZE,
            "chunk_overlap": RAG_CHUNK_OVERLAP,
            "top_k": RAG_TOP_K,
            "similarity_threshold": RAG_SIMILARITY_THRESHOLD
        },
        "clustering_config": {
            "num_clusters": NUM_CLUSTERS,
            "methods": CLUSTERING_METHODS,
            "default_method": DEFAULT_CLUSTERING_METHOD,
            "kmeans_params": {
                "n_init": KMEANS_N_INIT,
                "max_iter": KMEANS_MAX_ITER,
                "random_state": KMEANS_RANDOM_STATE
            },
            "dbscan_params": {
                "eps": DBSCAN_EPS,
                "min_samples": DBSCAN_MIN_SAMPLES
            },
            "hierarchical_params": {
                "linkage": HIERARCHICAL_LINKAGE,
                "affinity": HIERARCHICAL_AFFINITY
            }
        },
        "feature_config": {
            "scaling_method": FEATURE_SCALING_METHOD,
            "use_pca": USE_PCA,
            "pca_components": PCA_N_COMPONENTS,
            "patient_features": PATIENT_FEATURES
        },
        "risk_config": {
            "thresholds": RISK_THRESHOLDS,
            "weights": RISK_WEIGHTS
        },
        "deep_learning_config": DEEP_LEARNING_CONFIG,
        "random_forest_config": RANDOM_FOREST_CONFIG,
        "monitoring_plans": MONITORING_PLANS,
        "follow_up_schedules": FOLLOW_UP_SCHEDULES,
        "treatment_categories": TREATMENT_CATEGORIES,
        "knowledge_sources": KNOWLEDGE_SOURCES,
        "api_config": {
            "host": API_HOST,
            "port": API_PORT,
            "workers": API_WORKERS,
            "reload": API_RELOAD
        },
        "logging_config": {
            "level": LOG_LEVEL,
            "format": LOG_FORMAT,
            "file": LOG_FILE
        },
        "security_config": {
            "cors_origins": CORS_ORIGINS,
            "api_key_header": API_KEY_HEADER,
            "rate_limit": RATE_LIMIT_PER_MINUTE
        },
        "performance_config": {
            "cache_ttl": CACHE_TTL,
            "batch_size": BATCH_SIZE,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
        },
        "evaluation_config": {
            "metrics": EVALUATION_METRICS,
            "cv_folds": CROSS_VALIDATION_FOLDS,
            "test_size": TEST_SIZE
        },
        "validation_config": {
            "rules": DATA_VALIDATION_RULES
        },
        "alert_config": ALERT_THRESHOLDS,
        "notification_config": NOTIFICATION_CHANNELS
    }

if __name__ == "__main__":
    # Create directories and validate configuration
    create_directories()
    validate_config()
    
    # Print configuration summary
    config = get_config()
    print("Patient Stratification Configuration Summary:")
    print(f"- API Port: {config['api_config']['port']}")
    print(f"- Number of Clusters: {config['clustering_config']['num_clusters']}")
    print(f"- Default Clustering Method: {config['clustering_config']['default_method']}")
    print(f"- LLM Model: {config['llm_config']['model_name']}")
    print(f"- RAG Top-K: {config['rag_config']['top_k']}")
    print(f"- Data Paths: {list(config['data_paths'].keys())}") 