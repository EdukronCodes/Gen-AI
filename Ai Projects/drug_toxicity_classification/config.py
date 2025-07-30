"""
Configuration file for Drug Toxicity Classification System
Contains all API keys, paths, and system configurations
"""

import os
from typing import Optional

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./toxicity_chroma_db")

# Data Paths
TOXICITY_DATA_PATH = os.getenv("TOXICITY_DATA_PATH", "./data/toxicity")
RESEARCH_LITERATURE_PATH = os.getenv("RESEARCH_LITERATURE_PATH", "./data/literature")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "./models")
RESULTS_PATH = os.getenv("RESULTS_PATH", "./results")

# Model Configuration
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": 42
}

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,
    "random_state": 42
}

# CNN Configuration
CNN_PARAMS = {
    "input_size": 2048,
    "num_classes": 4,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
}

# GNN Configuration
GNN_PARAMS = {
    "num_node_features": 74,
    "num_classes": 4,
    "hidden_channels": 64,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
}

# RAG Configuration
RAG_PARAMS = {
    "top_k_results": 5,
    "similarity_threshold": 0.7,
    "chunk_size": 1000,
    "chunk_overlap": 200
}

# Toxicity Classes
TOXICITY_CLASSES = {
    "low": "Low toxicity - Generally safe for use",
    "moderate": "Moderate toxicity - Requires careful monitoring",
    "high": "High toxicity - Significant safety concerns",
    "very_high": "Very high toxicity - Avoid use"
}

# Molecular Descriptors
MOLECULAR_DESCRIPTORS = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
    'NumRotatableBonds', 'NumAromaticRings', 'FractionCsp3',
    'HeavyAtomCount', 'RingCount', 'HBA', 'HBD', 'PSA'
]

# Risk Factor Thresholds
RISK_THRESHOLDS = {
    "molecular_weight": 500,  # Da
    "logp": 5.0,
    "num_h_donors": 5,
    "num_h_acceptors": 10,
    "tpsa": 140,  # Angstroms squared
    "num_rotatable_bonds": 10,
    "num_aromatic_rings": 3
}

# Data Sources
TOXICITY_DATA_SOURCES = {
    "pubmed_toxicity": {
        "path": f"{RESEARCH_LITERATURE_PATH}/pubmed",
        "type": "research_articles",
        "update_frequency": "weekly"
    },
    "regulatory_guidelines": {
        "path": f"{RESEARCH_LITERATURE_PATH}/guidelines",
        "type": "regulatory_documents",
        "update_frequency": "monthly"
    },
    "drug_database": {
        "path": f"{TOXICITY_DATA_PATH}/drug_db",
        "type": "drug_information",
        "update_frequency": "daily"
    },
    "clinical_trials": {
        "path": f"{RESEARCH_LITERATURE_PATH}/clinical_trials",
        "type": "trial_data",
        "update_frequency": "weekly"
    },
    "safety_assessments": {
        "path": f"{RESEARCH_LITERATURE_PATH}/safety",
        "type": "safety_reports",
        "update_frequency": "monthly"
    }
}

# Model Performance Metrics
PERFORMANCE_METRICS = [
    "accuracy", "precision", "recall", "f1_score", 
    "roc_auc", "confusion_matrix", "classification_report"
]

# Validation Configuration
VALIDATION_PARAMS = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_state": 42,
    "stratify": True,
    "cross_validation_folds": 5
}

# Training Configuration
TRAINING_PARAMS = {
    "early_stopping_patience": 10,
    "learning_rate_scheduler": True,
    "model_checkpoint": True,
    "tensorboard_logging": True,
    "wandb_logging": False
}

# API Configuration
API_PARAMS = {
    "host": "0.0.0.0",
    "port": 8001,
    "debug": False,
    "reload": False,
    "workers": 1
}

# Logging Configuration
LOGGING_PARAMS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "./logs/toxicity_classifier.log",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# Security Configuration
SECURITY_PARAMS = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-here"),
    "access_token_expire_minutes": 30,
    "rate_limit": 100,  # requests per minute
    "cors_origins": ["*"]
}

# Database Configuration
DATABASE_PARAMS = {
    "url": os.getenv("DATABASE_URL", "sqlite:///./toxicity_classifier.db"),
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30
}

# Cache Configuration
CACHE_PARAMS = {
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "ttl": 3600,  # 1 hour
    "max_size": 1000
}

# File Upload Configuration
UPLOAD_PARAMS = {
    "max_file_size": 10485760,  # 10MB
    "allowed_extensions": [".csv", ".xlsx", ".txt"],
    "upload_path": "./uploads"
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
        TOXICITY_DATA_PATH,
        RESEARCH_LITERATURE_PATH,
        MODEL_SAVE_PATH,
        RESULTS_PATH,
        os.path.dirname(LOGGING_PARAMS["file"]),
        UPLOAD_PARAMS["upload_path"]
    ]
    
    for path in paths_to_create:
        os.makedirs(path, exist_ok=True)

# Utility Functions
def get_model_path(model_name: str) -> str:
    """Get the path for saving/loading a specific model"""
    return os.path.join(MODEL_SAVE_PATH, f"{model_name}.pkl")

def get_results_path(experiment_name: str) -> str:
    """Get the path for saving experiment results"""
    return os.path.join(RESULTS_PATH, f"{experiment_name}.json")

def get_data_source_path(source_name: str) -> str:
    """Get the path for a specific data source"""
    if source_name in TOXICITY_DATA_SOURCES:
        return TOXICITY_DATA_SOURCES[source_name]["path"]
    else:
        raise ValueError(f"Unknown data source: {source_name}")

# Initialize configuration
if __name__ == "__main__":
    validate_config()
    print("Configuration validated successfully!")
    print(f"Toxicity classes: {list(TOXICITY_CLASSES.keys())}")
    print(f"Molecular descriptors: {len(MOLECULAR_DESCRIPTORS)} descriptors")
    print(f"Data sources: {list(TOXICITY_DATA_SOURCES.keys())}") 