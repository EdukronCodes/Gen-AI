"""
Patient Stratification for Personalized Healthcare Interventions
A comprehensive system for clustering patients and generating personalized treatment recommendations with enhanced RAG integration
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import fastapi
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Enhanced RAG and Vector Search
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ChromaDB, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.retrievers import VectorStoreRetriever, BM25Retriever, EnsembleRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers.document_compressors import LLMChainExtractor

# Medical and Healthcare Libraries
import spacy
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

import chromadb
from chromadb.config import Settings
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import json
from typing import Union
import warnings
warnings.filterwarnings('ignore')

# Configuration
from config import (
    OPENAI_API_KEY,
    CHROMA_DB_PATH,
    LLM_MODEL_NAME,
    RAG_PARAMS,
    KNOWLEDGE_SOURCES,
    CLUSTERING_PARAMS,
    RISK_THRESHOLDS,
    MONITORING_PLANS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Patient Stratification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class PatientData:
    """Patient data structure for stratification"""
    patient_id: str
    age: int
    gender: str
    bmi: float
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    heart_rate: int
    cholesterol_total: float
    cholesterol_hdl: float
    cholesterol_ldl: float
    triglycerides: float
    blood_sugar: float
    smoking_status: str
    diabetes_status: str
    family_history: List[str]
    medications: List[str]
    lab_results: Dict[str, float]
    vital_signs: Dict[str, float]
    symptoms: List[str]
    diagnosis_history: List[str]
    clinical_context: Dict[str, Any]

@dataclass
class StratificationResult:
    """Result of patient stratification with enhanced RAG insights"""
    patient_id: str
    cluster_id: int
    cluster_label: str
    risk_level: str
    confidence_score: float
    similar_patients: List[str]
    treatment_recommendations: List[str]
    monitoring_plan: Dict[str, Any]
    follow_up_schedule: Dict[str, str]
    clinical_evidence: Dict[str, Any]
    research_insights: Dict[str, Any]

class HealthcareRAGSystem:
    """Enhanced RAG system for healthcare knowledge retrieval with clinical context awareness"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.bm25_retriever = None
        self.knowledge_base = {}
        self.retrieval_cache = {}
        self.clinical_databases = {}
        self.initialize_vector_store()
        self.load_healthcare_sources()
        self.setup_retrievers()
    
    def initialize_vector_store(self):
        """Initialize vector store with enhanced healthcare knowledge"""
        try:
            # Initialize ChromaDB with healthcare-specific settings
            self.vector_store = ChromaDB(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                client_settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Healthcare vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def setup_retrievers(self):
        """Setup multiple retrieval strategies for enhanced healthcare information retrieval"""
        try:
            # Setup BM25 retriever for keyword-based retrieval
            if hasattr(self.vector_store, 'docstore'):
                documents = list(self.vector_store.docstore.values())
                self.bm25_retriever = BM25Retriever.from_documents(documents)
            
            # Setup ensemble retriever combining vector and keyword search
            if self.bm25_retriever:
                vector_retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": RAG_PARAMS.get("top_k", 5)}
                )
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]
                )
            
            logger.info("Healthcare retrievers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
    
    def load_healthcare_sources(self):
        """Load comprehensive healthcare knowledge sources from various repositories"""
        try:
            # Load from local healthcare knowledge base
            if os.path.exists(KNOWLEDGE_SOURCES.get("local", "")):
                self.load_clinical_guidelines()
            
            # Load from PubMed clinical articles
            self.load_pubmed_clinical_articles()
            
            # Load from treatment protocols
            self.load_treatment_protocols()
            
            # Load from clinical trial data
            self.load_clinical_trial_data()
            
            # Load from medical guidelines
            self.load_medical_guidelines()
            
            # Load from patient outcome studies
            self.load_patient_outcome_studies()
            
            logger.info(f"Loaded {len(self.knowledge_base)} healthcare knowledge sources")
            
        except Exception as e:
            logger.error(f"Error loading healthcare sources: {e}")
    
    def load_clinical_guidelines(self):
        """Load clinical practice guidelines from various sources"""
        try:
            # Load from clinical guidelines databases
            guidelines = [
                {
                    "title": "Hypertension Management Guidelines",
                    "content": "Blood pressure should be monitored regularly with target values...",
                    "source": "American Heart Association",
                    "year": 2024,
                    "category": "cardiovascular"
                },
                {
                    "title": "Diabetes Management Guidelines",
                    "content": "Comprehensive diabetes care includes blood glucose monitoring...",
                    "source": "American Diabetes Association",
                    "year": 2024,
                    "category": "endocrinology"
                },
                {
                    "title": "Cholesterol Management Guidelines",
                    "content": "LDL cholesterol targets vary based on cardiovascular risk...",
                    "source": "American College of Cardiology",
                    "year": 2024,
                    "category": "cardiovascular"
                }
            ]
            
            for guideline in guidelines:
                self.knowledge_base[f"guideline_{guideline['source']}_{guideline['title']}"] = {
                    "content": guideline['content'],
                    "metadata": guideline,
                    "type": "guideline"
                }
                
        except Exception as e:
            logger.error(f"Error loading clinical guidelines: {e}")
    
    def load_pubmed_clinical_articles(self):
        """Load clinical research articles from PubMed"""
        try:
            # This would integrate with PubMed API
            # For now, we'll simulate with sample data
            clinical_articles = [
                {
                    "title": "Patient stratification in cardiovascular disease management",
                    "abstract": "This study examines the effectiveness of patient stratification...",
                    "authors": ["Dr. Martinez", "Dr. Rodriguez"],
                    "journal": "Journal of Clinical Cardiology",
                    "year": 2024,
                    "pmid": "12345678"
                },
                {
                    "title": "Personalized treatment approaches for diabetes patients",
                    "abstract": "Personalized medicine approaches show improved outcomes...",
                    "authors": ["Dr. Johnson", "Dr. Smith"],
                    "journal": "Diabetes Care",
                    "year": 2024,
                    "pmid": "87654321"
                }
            ]
            
            for article in clinical_articles:
                self.knowledge_base[f"pubmed_{article['pmid']}"] = {
                    "content": f"{article['title']}\n{article['abstract']}",
                    "metadata": article,
                    "type": "pubmed"
                }
                
        except Exception as e:
            logger.error(f"Error loading PubMed clinical articles: {e}")
    
    def load_treatment_protocols(self):
        """Load treatment protocols and care pathways"""
        try:
            # Load treatment protocols
            protocols = [
                {
                    "condition": "Hypertension",
                    "protocol": "Step 1: Lifestyle modifications\nStep 2: ACE inhibitors or ARBs\nStep 3: Add calcium channel blockers",
                    "risk_factors": ["Age > 65", "Diabetes", "Kidney disease"],
                    "monitoring": ["Blood pressure", "Kidney function", "Electrolytes"]
                },
                {
                    "condition": "Type 2 Diabetes",
                    "protocol": "Step 1: Metformin\nStep 2: Add sulfonylurea or DPP-4 inhibitor\nStep 3: Consider insulin therapy",
                    "risk_factors": ["Obesity", "Family history", "Sedentary lifestyle"],
                    "monitoring": ["HbA1c", "Blood glucose", "Kidney function"]
                }
            ]
            
            for protocol in protocols:
                self.knowledge_base[f"protocol_{protocol['condition']}"] = {
                    "content": f"Condition: {protocol['condition']}\nProtocol: {protocol['protocol']}",
                    "metadata": protocol,
                    "type": "protocol"
                }
                
        except Exception as e:
            logger.error(f"Error loading treatment protocols: {e}")
    
    def load_clinical_trial_data(self):
        """Load clinical trial outcomes and safety data"""
        try:
            # Load clinical trial data
            trial_data = [
                {
                    "trial_id": "NCT12345678",
                    "intervention": "New antihypertensive medication",
                    "outcomes": ["Blood pressure reduction", "Cardiovascular events"],
                    "patient_population": "High-risk hypertension patients",
                    "results": "Significant blood pressure reduction with good safety profile"
                }
            ]
            
            for trial in trial_data:
                self.knowledge_base[f"trial_{trial['trial_id']}"] = {
                    "content": f"Trial: {trial['trial_id']}\nIntervention: {trial['intervention']}\nResults: {trial['results']}",
                    "metadata": trial,
                    "type": "clinical_trial"
                }
                
        except Exception as e:
            logger.error(f"Error loading clinical trial data: {e}")
    
    def load_medical_guidelines(self):
        """Load medical specialty guidelines"""
        try:
            # Load specialty guidelines
            specialty_guidelines = [
                {
                    "specialty": "Cardiology",
                    "guideline": "Cardiovascular risk assessment and management",
                    "content": "Comprehensive cardiovascular risk assessment includes...",
                    "evidence_level": "A"
                },
                {
                    "specialty": "Endocrinology",
                    "guideline": "Diabetes management and complications",
                    "content": "Diabetes management requires multidisciplinary approach...",
                    "evidence_level": "A"
                }
            ]
            
            for guideline in specialty_guidelines:
                self.knowledge_base[f"specialty_{guideline['specialty']}"] = {
                    "content": guideline['content'],
                    "metadata": guideline,
                    "type": "specialty_guideline"
                }
                
        except Exception as e:
            logger.error(f"Error loading medical guidelines: {e}")
    
    def load_patient_outcome_studies(self):
        """Load patient outcome and quality of life studies"""
        try:
            # Load outcome studies
            outcome_studies = [
                {
                    "study_id": "OUT001",
                    "condition": "Hypertension",
                    "outcomes": ["Quality of life", "Medication adherence", "Blood pressure control"],
                    "findings": "Personalized treatment improves adherence and outcomes"
                }
            ]
            
            for study in outcome_studies:
                self.knowledge_base[f"outcome_{study['study_id']}"] = {
                    "content": f"Study: {study['study_id']}\nCondition: {study['condition']}\nFindings: {study['findings']}",
                    "metadata": study,
                    "type": "outcome_study"
                }
                
        except Exception as e:
            logger.error(f"Error loading patient outcome studies: {e}")
    
    def retrieve_healthcare_knowledge(self, query: str, patient_context: Dict[str, Any] = None, 
                                    top_k: int = 5, use_ensemble: bool = True) -> List[Dict]:
        """Enhanced retrieval with patient context awareness and multiple strategies"""
        try:
            # Check cache first
            cache_key = f"{query}_{str(patient_context)}_{top_k}_{use_ensemble}"
            if cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            # Enhance query with patient context
            if patient_context:
                enhanced_query = self.enhance_query_with_context(query, patient_context)
            else:
                enhanced_query = query
            
            retrieved_info = []
            
            if use_ensemble and hasattr(self, 'ensemble_retriever'):
                # Use ensemble retriever
                docs = self.ensemble_retriever.get_relevant_documents(enhanced_query)
            else:
                # Use vector store directly
                docs = self.vector_store.similarity_search(enhanced_query, k=top_k)
            
            # Process retrieved documents
            for doc in docs:
                info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": self.calculate_healthcare_relevance_score(enhanced_query, doc.page_content, patient_context),
                    "evidence_level": doc.metadata.get("evidence_level", "unknown")
                }
                retrieved_info.append(info)
            
            # Sort by relevance score
            retrieved_info.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Cache results
            self.retrieval_cache[cache_key] = retrieved_info[:top_k]
            
            logger.info(f"Retrieved {len(retrieved_info)} relevant documents for healthcare query: {query[:50]}...")
            return retrieved_info[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving healthcare knowledge: {e}")
            return []
    
    def enhance_query_with_context(self, query: str, patient_context: Dict[str, Any]) -> str:
        """Enhance query with patient-specific context"""
        try:
            enhanced_parts = [query]
            
            # Add age and gender context
            if 'age' in patient_context:
                enhanced_parts.append(f"age {patient_context['age']}")
            if 'gender' in patient_context:
                enhanced_parts.append(f"gender {patient_context['gender']}")
            
            # Add medical conditions
            if 'diagnosis_history' in patient_context:
                conditions = patient_context['diagnosis_history']
                if conditions:
                    enhanced_parts.append(f"conditions {' '.join(conditions)}")
            
            # Add risk factors
            if 'risk_factors' in patient_context:
                risk_factors = patient_context['risk_factors']
                if risk_factors:
                    enhanced_parts.append(f"risk factors {' '.join(risk_factors)}")
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query with context: {e}")
            return query
    
    def calculate_healthcare_relevance_score(self, query: str, content: str, patient_context: Dict[str, Any] = None) -> float:
        """Calculate relevance score for healthcare content with patient context"""
        try:
            # Base relevance calculation
            query_words = set(query.lower().split())
            content_words = content.lower().split()
            
            # Calculate word overlap
            overlap = len(query_words.intersection(set(content_words)))
            total_query_words = len(query_words)
            
            if total_query_words == 0:
                return 0.0
            
            relevance = overlap / total_query_words
            
            # Boost score for healthcare-related terms
            healthcare_terms = [
                "treatment", "therapy", "medication", "diagnosis", "symptom",
                "guideline", "protocol", "clinical", "patient", "outcome",
                "risk", "prevention", "management", "care", "health"
            ]
            healthcare_boost = sum(1 for term in healthcare_terms if term in content.lower())
            relevance += healthcare_boost * 0.1
            
            # Boost for patient-specific context
            if patient_context:
                # Boost for age-appropriate content
                if 'age' in patient_context:
                    age = patient_context['age']
                    if age > 65 and "elderly" in content.lower():
                        relevance += 0.1
                    elif age < 18 and "pediatric" in content.lower():
                        relevance += 0.1
                
                # Boost for condition-specific content
                if 'diagnosis_history' in patient_context:
                    conditions = patient_context['diagnosis_history']
                    for condition in conditions:
                        if condition.lower() in content.lower():
                            relevance += 0.15
            
            # Boost for evidence-based content
            evidence_indicators = ["study", "trial", "research", "evidence", "guideline"]
            evidence_boost = sum(1 for indicator in evidence_indicators if indicator in content.lower())
            relevance += evidence_boost * 0.05
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating healthcare relevance score: {e}")
            return 0.0
    
    def generate_treatment_recommendations(self, patient_data: PatientData, cluster_info: Dict[str, Any]) -> List[str]:
        """Generate evidence-based treatment recommendations using RAG"""
        try:
            recommendations = []
            
            # Build patient context for RAG query
            patient_context = {
                "age": patient_data.age,
                "gender": patient_data.gender,
                "diagnosis_history": patient_data.diagnosis_history,
                "risk_factors": self.extract_risk_factors(patient_data),
                "cluster_characteristics": cluster_info.get("description", "")
            }
            
            # Query for treatment guidelines
            treatment_query = f"treatment recommendations for {cluster_info.get('label', 'patient cluster')}"
            treatment_info = self.retrieve_healthcare_knowledge(treatment_query, patient_context, top_k=3)
            
            # Generate recommendations based on retrieved information
            for info in treatment_info:
                if "treatment" in info["content"].lower() or "therapy" in info["content"].lower():
                    # Extract treatment recommendations from content
                    treatment_suggestions = self.extract_treatment_suggestions(info["content"])
                    recommendations.extend(treatment_suggestions)
            
            # Add cluster-specific recommendations
            if cluster_info.get("risk_level") == "High":
                recommendations.append("Consider intensive monitoring and frequent follow-up")
                recommendations.append("Implement comprehensive risk factor management")
            
            # Add evidence-based recommendations
            evidence_query = f"evidence-based interventions for {cluster_info.get('label', 'patient group')}"
            evidence_info = self.retrieve_healthcare_knowledge(evidence_query, patient_context, top_k=2)
            
            for info in evidence_info:
                if "evidence" in info["content"].lower() or "study" in info["content"].lower():
                    evidence_suggestions = self.extract_evidence_based_recommendations(info["content"])
                    recommendations.extend(evidence_suggestions)
            
            return list(set(recommendations))[:10]  # Remove duplicates and limit
            
        except Exception as e:
            logger.error(f"Error generating treatment recommendations: {e}")
            return ["Consult with healthcare provider for personalized treatment plan"]
    
    def extract_risk_factors(self, patient_data: PatientData) -> List[str]:
        """Extract risk factors from patient data"""
        risk_factors = []
        
        if patient_data.smoking_status == "Yes":
            risk_factors.append("smoking")
        if patient_data.diabetes_status == "Yes":
            risk_factors.append("diabetes")
        if patient_data.bmi > 30:
            risk_factors.append("obesity")
        if patient_data.blood_pressure_systolic > 140:
            risk_factors.append("hypertension")
        if patient_data.cholesterol_ldl > 130:
            risk_factors.append("high cholesterol")
        
        return risk_factors
    
    def extract_treatment_suggestions(self, content: str) -> List[str]:
        """Extract treatment suggestions from content"""
        suggestions = []
        
        # Simple extraction based on keywords
        treatment_keywords = ["recommend", "suggest", "consider", "prescribe", "administer"]
        
        sentences = content.split(".")
        for sentence in sentences:
            for keyword in treatment_keywords:
                if keyword in sentence.lower():
                    suggestions.append(sentence.strip())
                    break
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def extract_evidence_based_recommendations(self, content: str) -> List[str]:
        """Extract evidence-based recommendations from content"""
        recommendations = []
        
        # Look for evidence-based language
        evidence_indicators = ["study shows", "research indicates", "evidence suggests", "clinical trial"]
        
        sentences = content.split(".")
        for sentence in sentences:
            for indicator in evidence_indicators:
                if indicator in sentence.lower():
                    recommendations.append(sentence.strip())
                    break
        
        return recommendations[:3]  # Limit to 3 recommendations

class PatientDataset(Dataset):
    """Custom PyTorch dataset for patient data"""
    
    def __init__(self, patient_data: List[PatientData], features: List[str]):
        self.patient_data = patient_data
        self.features = features
        self.scaler = StandardScaler()
        self.processed_data = self.preprocess_data()
    
    def preprocess_data(self) -> np.ndarray:
        """Preprocess patient data for clustering"""
        processed_data = []
        
        for patient in self.patient_data:
            feature_vector = []
            
            # Numerical features
            feature_vector.extend([
                patient.age,
                patient.bmi,
                patient.blood_pressure_systolic,
                patient.blood_pressure_diastolic,
                patient.heart_rate,
                patient.cholesterol_total,
                patient.cholesterol_hdl,
                patient.cholesterol_ldl,
                patient.triglycerides,
                patient.blood_sugar
            ])
            
            # Categorical features (one-hot encoded)
            gender_encoded = [1 if patient.gender == 'M' else 0]
            smoking_encoded = [1 if patient.smoking_status == 'Yes' else 0]
            diabetes_encoded = [1 if patient.diabetes_status == 'Yes' else 0]
            
            feature_vector.extend(gender_encoded + smoking_encoded + diabetes_encoded)
            
            processed_data.append(feature_vector)
        
        # Scale the data
        processed_data = np.array(processed_data)
        self.scaler.fit(processed_data)
        processed_data = self.scaler.transform(processed_data)
        
        return processed_data
    
    def __len__(self):
        return len(self.patient_data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.processed_data[idx])

class PatientStratificationModel(nn.Module):
    """Deep learning model for patient stratification"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_clusters: int):
        super(PatientStratificationModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.cluster_layer = nn.Linear(prev_dim, num_clusters)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        cluster_assignments = torch.softmax(self.cluster_layer(features), dim=1)
        return features, cluster_assignments

class PatientStratificationSystem:
    """Main patient stratification system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = HealthcareRAGSystem(config)
        self.scaler = StandardScaler()
        self.clustering_models = {}
        self.stratification_model = None
        self.cluster_descriptions = {}
        self.risk_assessment_model = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all models for patient stratification"""
        try:
            # Initialize clustering models
            self.clustering_models = {
                'kmeans': KMeans(
                    n_clusters=self.config.get('num_clusters', 5),
                    random_state=42,
                    n_init=10
                ),
                'dbscan': DBSCAN(
                    eps=self.config.get('dbscan_eps', 0.5),
                    min_samples=self.config.get('dbscan_min_samples', 5)
                ),
                'hierarchical': AgglomerativeClustering(
                    n_clusters=self.config.get('num_clusters', 5),
                    linkage='ward'
                )
            }
            
            # Initialize deep learning model
            input_dim = self.config.get('input_dim', 13)  # Number of features
            hidden_dims = self.config.get('hidden_dims', [64, 32, 16])
            num_clusters = self.config.get('num_clusters', 5)
            
            self.stratification_model = PatientStratificationModel(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_clusters=num_clusters
            )
            
            # Initialize risk assessment model
            self.risk_assessment_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            logger.info("Patient stratification models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def prepare_patient_data(self, patients: List[PatientData]) -> np.ndarray:
        """Prepare patient data for clustering"""
        try:
            processed_data = []
            
            for patient in patients:
                feature_vector = [
                    patient.age,
                    patient.bmi,
                    patient.blood_pressure_systolic,
                    patient.blood_pressure_diastolic,
                    patient.heart_rate,
                    patient.cholesterol_total,
                    patient.cholesterol_hdl,
                    patient.cholesterol_ldl,
                    patient.triglycerides,
                    patient.blood_sugar,
                    1 if patient.gender == 'M' else 0,
                    1 if patient.smoking_status == 'Yes' else 0,
                    1 if patient.diabetes_status == 'Yes' else 0
                ]
                processed_data.append(feature_vector)
            
            processed_data = np.array(processed_data)
            self.scaler.fit(processed_data)
            processed_data = self.scaler.transform(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preparing patient data: {e}")
            raise
    
    def perform_clustering(self, patient_data: np.ndarray, method: str = 'kmeans') -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform clustering on patient data"""
        try:
            if method not in self.clustering_models:
                raise ValueError(f"Unknown clustering method: {method}")
            
            model = self.clustering_models[method]
            cluster_labels = model.fit_predict(patient_data)
            
            # Calculate clustering metrics
            metrics = {}
            if len(set(cluster_labels)) > 1:
                try:
                    metrics['silhouette_score'] = silhouette_score(patient_data, cluster_labels)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(patient_data, cluster_labels)
                except:
                    pass
            
            # Generate cluster descriptions
            cluster_info = self.generate_cluster_descriptions(patient_data, cluster_labels)
            
            return cluster_labels, cluster_info
            
        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
            raise
    
    def generate_cluster_descriptions(self, patient_data: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Generate descriptions for each cluster"""
        try:
            cluster_info = {}
            unique_clusters = set(cluster_labels)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Noise points in DBSCAN
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_data = patient_data[cluster_mask]
                
                # Calculate cluster statistics
                cluster_stats = {
                    'size': int(np.sum(cluster_mask)),
                    'mean_age': float(np.mean(cluster_data[:, 0])),
                    'mean_bmi': float(np.mean(cluster_data[:, 1])),
                    'mean_bp_systolic': float(np.mean(cluster_data[:, 2])),
                    'mean_bp_diastolic': float(np.mean(cluster_data[:, 3])),
                    'mean_heart_rate': float(np.mean(cluster_data[:, 4])),
                    'mean_cholesterol': float(np.mean(cluster_data[:, 5])),
                    'mean_blood_sugar': float(np.mean(cluster_data[:, 10])),
                    'male_percentage': float(np.mean(cluster_data[:, 10]) * 100),
                    'smoker_percentage': float(np.mean(cluster_data[:, 11]) * 100),
                    'diabetic_percentage': float(np.mean(cluster_data[:, 12]) * 100)
                }
                
                # Generate risk level
                risk_score = self.calculate_risk_score(cluster_stats)
                cluster_stats['risk_level'] = self.classify_risk_level(risk_score)
                
                # Generate description
                description = self.generate_cluster_description(cluster_stats)
                cluster_stats['description'] = description
                
                cluster_info[f"cluster_{cluster_id}"] = cluster_stats
            
            return cluster_info
            
        except Exception as e:
            logger.error(f"Error generating cluster descriptions: {e}")
            return {}
    
    def calculate_risk_score(self, cluster_stats: Dict[str, float]) -> float:
        """Calculate risk score for a cluster"""
        try:
            risk_factors = []
            
            # Age risk (higher risk for older patients)
            if cluster_stats['mean_age'] > 65:
                risk_factors.append(2.0)
            elif cluster_stats['mean_age'] > 50:
                risk_factors.append(1.5)
            else:
                risk_factors.append(1.0)
            
            # BMI risk
            if cluster_stats['mean_bmi'] > 30:
                risk_factors.append(2.0)
            elif cluster_stats['mean_bmi'] > 25:
                risk_factors.append(1.5)
            else:
                risk_factors.append(1.0)
            
            # Blood pressure risk
            if cluster_stats['mean_bp_systolic'] > 140 or cluster_stats['mean_bp_diastolic'] > 90:
                risk_factors.append(2.0)
            else:
                risk_factors.append(1.0)
            
            # Cholesterol risk
            if cluster_stats['mean_cholesterol'] > 200:
                risk_factors.append(1.5)
            else:
                risk_factors.append(1.0)
            
            # Blood sugar risk
            if cluster_stats['mean_blood_sugar'] > 126:
                risk_factors.append(2.0)
            elif cluster_stats['mean_blood_sugar'] > 100:
                risk_factors.append(1.5)
            else:
                risk_factors.append(1.0)
            
            # Lifestyle risk factors
            if cluster_stats['smoker_percentage'] > 50:
                risk_factors.append(2.0)
            elif cluster_stats['smoker_percentage'] > 20:
                risk_factors.append(1.5)
            else:
                risk_factors.append(1.0)
            
            if cluster_stats['diabetic_percentage'] > 30:
                risk_factors.append(2.0)
            elif cluster_stats['diabetic_percentage'] > 10:
                risk_factors.append(1.5)
            else:
                risk_factors.append(1.0)
            
            # Calculate average risk score
            return np.mean(risk_factors)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 1.0
    
    def classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on risk score"""
        if risk_score >= 1.8:
            return "High"
        elif risk_score >= 1.4:
            return "Medium"
        else:
            return "Low"
    
    def generate_cluster_description(self, cluster_stats: Dict[str, float]) -> str:
        """Generate human-readable description of cluster"""
        try:
            description_parts = []
            
            # Age description
            if cluster_stats['mean_age'] > 65:
                description_parts.append("elderly patients")
            elif cluster_stats['mean_age'] > 50:
                description_parts.append("middle-aged patients")
            else:
                description_parts.append("younger patients")
            
            # BMI description
            if cluster_stats['mean_bmi'] > 30:
                description_parts.append("with obesity")
            elif cluster_stats['mean_bmi'] > 25:
                description_parts.append("with overweight")
            else:
                description_parts.append("with normal weight")
            
            # Blood pressure description
            if cluster_stats['mean_bp_systolic'] > 140 or cluster_stats['mean_bp_diastolic'] > 90:
                description_parts.append("with hypertension")
            
            # Diabetes description
            if cluster_stats['diabetic_percentage'] > 30:
                description_parts.append("with high diabetes prevalence")
            elif cluster_stats['diabetic_percentage'] > 10:
                description_parts.append("with moderate diabetes prevalence")
            
            # Smoking description
            if cluster_stats['smoker_percentage'] > 50:
                description_parts.append("with high smoking rates")
            elif cluster_stats['smoker_percentage'] > 20:
                description_parts.append("with moderate smoking rates")
            
            # Risk level
            description_parts.append(f"({cluster_stats['risk_level']} risk)")
            
            return " ".join(description_parts)
            
        except Exception as e:
            logger.error(f"Error generating cluster description: {e}")
            return "Patient cluster"
    
    def stratify_patient(self, patient: PatientData, clustering_method: str = 'kmeans') -> StratificationResult:
        """Stratify a single patient"""
        try:
            # Prepare patient data
            patient_data = self.prepare_patient_data([patient])
            
            # Perform clustering
            cluster_labels, cluster_info = self.perform_clustering(patient_data, clustering_method)
            cluster_id = int(cluster_labels[0])
            
            # Get cluster information
            cluster_key = f"cluster_{cluster_id}"
            if cluster_key in cluster_info:
                cluster_data = cluster_info[cluster_key]
                cluster_label = cluster_data.get('description', f'Cluster {cluster_id}')
                risk_level = cluster_data.get('risk_level', 'Medium')
            else:
                cluster_label = f'Cluster {cluster_id}'
                risk_level = 'Medium'
            
            # Calculate confidence score
            confidence_score = self.calculate_confidence_score(patient_data[0], cluster_data)
            
            # Find similar patients
            similar_patients = self.find_similar_patients(patient_data[0], cluster_id)
            
            # Generate treatment recommendations using RAG
            treatment_recommendations = self.rag_system.generate_treatment_recommendations(
                patient, cluster_data
            )
            
            # Generate monitoring plan
            monitoring_plan = self.generate_monitoring_plan(patient, risk_level)
            
            # Generate follow-up schedule
            follow_up_schedule = self.generate_follow_up_schedule(risk_level)
            
            return StratificationResult(
                patient_id=patient.patient_id,
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                risk_level=risk_level,
                confidence_score=confidence_score,
                similar_patients=similar_patients,
                treatment_recommendations=treatment_recommendations,
                monitoring_plan=monitoring_plan,
                follow_up_schedule=follow_up_schedule
            )
            
        except Exception as e:
            logger.error(f"Error stratifying patient: {e}")
            raise
    
    def calculate_confidence_score(self, patient_features: np.ndarray, cluster_data: Dict[str, Any]) -> float:
        """Calculate confidence score for patient stratification"""
        try:
            # Calculate distance from cluster center
            cluster_center = np.array([
                cluster_data.get('mean_age', 50),
                cluster_data.get('mean_bmi', 25),
                cluster_data.get('mean_bp_systolic', 120),
                cluster_data.get('mean_bp_diastolic', 80),
                cluster_data.get('mean_heart_rate', 70),
                cluster_data.get('mean_cholesterol', 200),
                cluster_data.get('mean_cholesterol_hdl', 50),
                cluster_data.get('mean_cholesterol_ldl', 100),
                cluster_data.get('mean_triglycerides', 150),
                cluster_data.get('mean_blood_sugar', 100),
                0.5,  # gender
                0.2,  # smoking
                0.1   # diabetes
            ])
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(patient_features - cluster_center)
            
            # Convert distance to confidence score (0-1)
            confidence = max(0, 1 - distance / 10)
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def find_similar_patients(self, patient_features: np.ndarray, cluster_id: int) -> List[str]:
        """Find similar patients in the same cluster"""
        try:
            # This would typically query a database of patients
            # For now, return a placeholder
            return [f"similar_patient_{i}" for i in range(3)]
            
        except Exception as e:
            logger.error(f"Error finding similar patients: {e}")
            return []
    
    def generate_monitoring_plan(self, patient: PatientData, risk_level: str) -> Dict[str, Any]:
        """Generate monitoring plan based on risk level"""
        try:
            if risk_level == "High":
                return {
                    "blood_pressure_monitoring": "Daily",
                    "blood_sugar_monitoring": "Daily",
                    "weight_monitoring": "Weekly",
                    "cholesterol_testing": "Monthly",
                    "doctor_visits": "Monthly",
                    "specialist_consultation": "Quarterly"
                }
            elif risk_level == "Medium":
                return {
                    "blood_pressure_monitoring": "Weekly",
                    "blood_sugar_monitoring": "Weekly",
                    "weight_monitoring": "Monthly",
                    "cholesterol_testing": "Quarterly",
                    "doctor_visits": "Quarterly",
                    "specialist_consultation": "As needed"
                }
            else:
                return {
                    "blood_pressure_monitoring": "Monthly",
                    "blood_sugar_monitoring": "Monthly",
                    "weight_monitoring": "Quarterly",
                    "cholesterol_testing": "Annually",
                    "doctor_visits": "Annually",
                    "specialist_consultation": "As needed"
                }
                
        except Exception as e:
            logger.error(f"Error generating monitoring plan: {e}")
            return {}
    
    def generate_follow_up_schedule(self, risk_level: str) -> Dict[str, str]:
        """Generate follow-up schedule based on risk level"""
        try:
            if risk_level == "High":
                return {
                    "next_appointment": "1 week",
                    "blood_work": "2 weeks",
                    "specialist_review": "1 month",
                    "comprehensive_assessment": "3 months"
                }
            elif risk_level == "Medium":
                return {
                    "next_appointment": "1 month",
                    "blood_work": "3 months",
                    "specialist_review": "6 months",
                    "comprehensive_assessment": "1 year"
                }
            else:
                return {
                    "next_appointment": "3 months",
                    "blood_work": "6 months",
                    "specialist_review": "1 year",
                    "comprehensive_assessment": "1 year"
                }
                
        except Exception as e:
            logger.error(f"Error generating follow-up schedule: {e}")
            return {}

# Pydantic models for API
class PatientDataRequest(BaseModel):
    patient_id: str
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., regex="^(M|F)$")
    bmi: float = Field(..., ge=10, le=100)
    blood_pressure_systolic: int = Field(..., ge=70, le=300)
    blood_pressure_diastolic: int = Field(..., ge=40, le=200)
    heart_rate: int = Field(..., ge=40, le=200)
    cholesterol_total: float = Field(..., ge=50, le=500)
    cholesterol_hdl: float = Field(..., ge=20, le=100)
    cholesterol_ldl: float = Field(..., ge=20, le=300)
    triglycerides: float = Field(..., ge=50, le=1000)
    blood_sugar: float = Field(..., ge=50, le=500)
    smoking_status: str = Field(..., regex="^(Yes|No)$")
    diabetes_status: str = Field(..., regex="^(Yes|No)$")
    family_history: List[str] = []
    medications: List[str] = []
    lab_results: Dict[str, float] = {}
    vital_signs: Dict[str, float] = {}
    symptoms: List[str] = []
    diagnosis_history: List[str] = []

class StratificationResponse(BaseModel):
    patient_id: str
    cluster_id: int
    cluster_label: str
    risk_level: str
    confidence_score: float
    similar_patients: List[str]
    treatment_recommendations: List[str]
    monitoring_plan: Dict[str, Any]
    follow_up_schedule: Dict[str, str]
    timestamp: datetime

class BatchStratificationRequest(BaseModel):
    patients: List[PatientDataRequest]
    clustering_method: str = Field(default="kmeans", regex="^(kmeans|dbscan|hierarchical)$")

class BatchStratificationResponse(BaseModel):
    results: List[StratificationResponse]
    clustering_metrics: Dict[str, float]
    timestamp: datetime

# Initialize the system
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
    "chroma_db_path": "./patient_stratification_chroma_db",
    "llm_model_name": "gpt-4",
    "num_clusters": 5,
    "input_dim": 13,
    "hidden_dims": [64, 32, 16],
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 5,
    "clinical_guidelines_path": "./data/clinical_guidelines",
    "treatment_protocols_path": "./data/treatment_protocols"
}

stratification_system = PatientStratificationSystem(config)

@app.post("/stratify", response_model=StratificationResponse)
async def stratify_patient(request: PatientDataRequest):
    """Stratify a single patient"""
    try:
        # Convert request to PatientData
        patient = PatientData(
            patient_id=request.patient_id,
            age=request.age,
            gender=request.gender,
            bmi=request.bmi,
            blood_pressure_systolic=request.blood_pressure_systolic,
            blood_pressure_diastolic=request.blood_pressure_diastolic,
            heart_rate=request.heart_rate,
            cholesterol_total=request.cholesterol_total,
            cholesterol_hdl=request.cholesterol_hdl,
            cholesterol_ldl=request.cholesterol_ldl,
            triglycerides=request.triglycerides,
            blood_sugar=request.blood_sugar,
            smoking_status=request.smoking_status,
            diabetes_status=request.diabetes_status,
            family_history=request.family_history,
            medications=request.medications,
            lab_results=request.lab_results,
            vital_signs=request.vital_signs,
            symptoms=request.symptoms,
            diagnosis_history=request.diagnosis_history
        )
        
        # Perform stratification
        result = stratification_system.stratify_patient(patient)
        
        # Convert to response format
        response = StratificationResponse(
            patient_id=result.patient_id,
            cluster_id=result.cluster_id,
            cluster_label=result.cluster_label,
            risk_level=result.risk_level,
            confidence_score=result.confidence_score,
            similar_patients=result.similar_patients,
            treatment_recommendations=result.treatment_recommendations,
            monitoring_plan=result.monitoring_plan,
            follow_up_schedule=result.follow_up_schedule,
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in stratify endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_stratify", response_model=BatchStratificationResponse)
async def batch_stratify_patients(request: BatchStratificationRequest):
    """Stratify multiple patients"""
    try:
        # Convert requests to PatientData objects
        patients = []
        for patient_req in request.patients:
            patient = PatientData(
                patient_id=patient_req.patient_id,
                age=patient_req.age,
                gender=patient_req.gender,
                bmi=patient_req.bmi,
                blood_pressure_systolic=patient_req.blood_pressure_systolic,
                blood_pressure_diastolic=patient_req.blood_pressure_diastolic,
                heart_rate=patient_req.heart_rate,
                cholesterol_total=patient_req.cholesterol_total,
                cholesterol_hdl=patient_req.cholesterol_hdl,
                cholesterol_ldl=patient_req.cholesterol_ldl,
                triglycerides=patient_req.triglycerides,
                blood_sugar=patient_req.blood_sugar,
                smoking_status=patient_req.smoking_status,
                diabetes_status=patient_req.diabetes_status,
                family_history=patient_req.family_history,
                medications=patient_req.medications,
                lab_results=patient_req.lab_results,
                vital_signs=patient_req.vital_signs,
                symptoms=patient_req.symptoms,
                diagnosis_history=patient_req.diagnosis_history
            )
            patients.append(patient)
        
        # Prepare data for clustering
        patient_data = stratification_system.prepare_patient_data(patients)
        
        # Perform clustering
        cluster_labels, cluster_info = stratification_system.perform_clustering(
            patient_data, request.clustering_method
        )
        
        # Generate results for each patient
        results = []
        for i, patient in enumerate(patients):
            cluster_id = int(cluster_labels[i])
            cluster_key = f"cluster_{cluster_id}"
            
            if cluster_key in cluster_info:
                cluster_data = cluster_info[cluster_key]
                cluster_label = cluster_data.get('description', f'Cluster {cluster_id}')
                risk_level = cluster_data.get('risk_level', 'Medium')
            else:
                cluster_label = f'Cluster {cluster_id}'
                risk_level = 'Medium'
            
            # Calculate confidence score
            confidence_score = stratification_system.calculate_confidence_score(
                patient_data[i], cluster_data
            )
            
            # Generate treatment recommendations
            treatment_recommendations = stratification_system.rag_system.generate_treatment_recommendations(
                patient, cluster_data
            )
            
            # Generate monitoring plan and follow-up schedule
            monitoring_plan = stratification_system.generate_monitoring_plan(patient, risk_level)
            follow_up_schedule = stratification_system.generate_follow_up_schedule(risk_level)
            
            result = StratificationResponse(
                patient_id=patient.patient_id,
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                risk_level=risk_level,
                confidence_score=confidence_score,
                similar_patients=[f"similar_patient_{j}" for j in range(3)],
                treatment_recommendations=treatment_recommendations,
                monitoring_plan=monitoring_plan,
                follow_up_schedule=follow_up_schedule,
                timestamp=datetime.now()
            )
            results.append(result)
        
        # Calculate clustering metrics
        clustering_metrics = {}
        if len(set(cluster_labels)) > 1:
            try:
                clustering_metrics['silhouette_score'] = silhouette_score(patient_data, cluster_labels)
                clustering_metrics['calinski_harabasz_score'] = calinski_harabasz_score(patient_data, cluster_labels)
            except:
                pass
        
        return BatchStratificationResponse(
            results=results,
            clustering_metrics=clustering_metrics,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in batch stratify endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "Patient Stratification API"
    }

@app.get("/clusters/info")
async def get_cluster_info():
    """Get information about current clusters"""
    try:
        return {
            "clustering_methods": list(stratification_system.clustering_models.keys()),
            "num_clusters": config.get("num_clusters", 5),
            "available_features": [
                "age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic",
                "heart_rate", "cholesterol_total", "cholesterol_hdl", "cholesterol_ldl",
                "triglycerides", "blood_sugar", "gender", "smoking_status", "diabetes_status"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting cluster info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 