"""
Health Insurance Fraud Detection System
A comprehensive system for detecting fraudulent insurance claims using ML and RAG
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import fastapi
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ChromaDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain.retrievers import VectorStoreRetriever, BM25Retriever, EnsembleRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.extractors import LLMChainExtractor
import chromadb
from chromadb.config import Settings
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import json
from typing import Union
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Health Insurance Fraud Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class InsuranceClaim:
    """Insurance claim data structure"""
    claim_id: str
    patient_id: str
    provider_id: str
    claim_date: datetime
    service_date: datetime
    diagnosis_codes: List[str]
    procedure_codes: List[str]
    billed_amount: float
    allowed_amount: float
    paid_amount: float
    provider_type: str
    service_type: str
    claim_type: str
    patient_demographics: Dict[str, Any]
    provider_info: Dict[str, Any]
    claim_details: Dict[str, Any]
    fraud_context: Optional[Dict[str, Any]] = None
    regulatory_context: Optional[str] = None

@dataclass
class FraudDetectionResult:
    """Result of fraud detection analysis"""
    claim_id: str
    fraud_probability: float
    fraud_score: float
    risk_level: str
    fraud_indicators: List[str]
    suspicious_patterns: List[str]
    recommendation: str
    confidence_score: float
    model_used: str
    analysis_timestamp: datetime
    regulatory_evidence: Optional[List[str]] = None
    fraud_analysis: Optional[Dict[str, str]] = None

class FraudDetectionRAGSystem:
    """Enhanced RAG system for fraud detection knowledge retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.get("openai_api_key")
        )
        self.vector_store = None
        self.retriever = None
        self.bm25_retriever = None
        self.retrieval_cache = {}
        self.fraud_databases = {}
        self.llm = ChatOpenAI(
            model_name=config.get("llm_model_name", "gpt-4"),
            temperature=0.1,
            openai_api_key=config.get("openai_api_key")
        )
        self.setup_retrievers()
        self.initialize_vector_store()
    
    def setup_retrievers(self):
        """Setup multiple retrieval strategies"""
        try:
            # Initialize BM25 retriever for keyword-based retrieval
            self.bm25_retriever = BM25Retriever(
                k=5,
                fetch_k=10
            )
            
            logger.info("Fraud detection retrievers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
    
    def initialize_vector_store(self):
        """Initialize the vector store with fraud detection documents"""
        try:
            # Initialize ChromaDB with settings
            chroma_client = chromadb.PersistentClient(
                path=self.config.get("chroma_db_path", "./fraud_detection_chroma_db"),
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Load fraud detection documents
            self.load_fraud_sources()
            
            # Create vector store
            self.vector_store = ChromaDB(
                client=chroma_client,
                collection_name="fraud_knowledge",
                embedding_function=self.embeddings
            )
            
            self.retriever = VectorStoreRetriever(
                vectorstore=self.vector_store,
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Setup ensemble retriever
            if self.bm25_retriever and self.retriever:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]
                )
            
            logger.info("Fraud detection RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def load_fraud_sources(self):
        """Load comprehensive fraud detection knowledge sources"""
        try:
            # Load fraud patterns and indicators
            self.load_fraud_patterns()
            
            # Load regulatory guidelines
            self.load_regulatory_guidelines()
            
            # Load case studies
            self.load_case_studies()
            
            # Load compliance documents
            self.load_compliance_documents()
            
            # Load industry reports
            self.load_industry_reports()
            
            # Load legal precedents
            self.load_legal_precedents()
            
            # Load fraud investigation guides
            self.load_investigation_guides()
            
            # Load risk assessment frameworks
            self.load_risk_frameworks()
            
            logger.info("Fraud detection knowledge sources loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fraud sources: {e}")
    
    def load_fraud_patterns(self):
        """Load fraud patterns and indicators"""
        try:
            patterns_path = self.config.get("fraud_patterns_path", "./data/fraud_patterns")
            if os.path.exists(patterns_path):
                loader = DirectoryLoader(
                    patterns_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                patterns_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                patterns_chunks = text_splitter.split_documents(patterns_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(patterns_chunks)
                    self.fraud_databases['fraud_patterns'] = patterns_chunks
                    
        except Exception as e:
            logger.error(f"Error loading fraud patterns: {e}")
    
    def load_regulatory_guidelines(self):
        """Load regulatory guidelines"""
        try:
            regulations_path = self.config.get("regulations_path", "./data/regulations")
            if os.path.exists(regulations_path):
                loader = DirectoryLoader(
                    regulations_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                regulations_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                regulations_chunks = text_splitter.split_documents(regulations_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(regulations_chunks)
                    self.fraud_databases['regulations'] = regulations_chunks
                    
        except Exception as e:
            logger.error(f"Error loading regulatory guidelines: {e}")
    
    def load_case_studies(self):
        """Load fraud case studies"""
        try:
            cases_path = self.config.get("case_studies_path", "./data/case_studies")
            if os.path.exists(cases_path):
                loader = DirectoryLoader(
                    cases_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                cases_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                cases_chunks = text_splitter.split_documents(cases_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(cases_chunks)
                    self.fraud_databases['case_studies'] = cases_chunks
                    
        except Exception as e:
            logger.error(f"Error loading case studies: {e}")
    
    def load_compliance_documents(self):
        """Load compliance documents"""
        try:
            compliance_path = self.config.get("compliance_path", "./data/compliance")
            if os.path.exists(compliance_path):
                loader = DirectoryLoader(
                    compliance_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                compliance_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                compliance_chunks = text_splitter.split_documents(compliance_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(compliance_chunks)
                    self.fraud_databases['compliance'] = compliance_chunks
                    
        except Exception as e:
            logger.error(f"Error loading compliance documents: {e}")
    
    def load_industry_reports(self):
        """Load industry fraud reports"""
        try:
            reports_path = self.config.get("industry_reports_path", "./data/industry_reports")
            if os.path.exists(reports_path):
                loader = DirectoryLoader(
                    reports_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                reports_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                reports_chunks = text_splitter.split_documents(reports_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(reports_chunks)
                    self.fraud_databases['industry_reports'] = reports_chunks
                    
        except Exception as e:
            logger.error(f"Error loading industry reports: {e}")
    
    def load_legal_precedents(self):
        """Load legal precedents and court cases"""
        try:
            legal_path = self.config.get("legal_precedents_path", "./data/legal_precedents")
            if os.path.exists(legal_path):
                loader = DirectoryLoader(
                    legal_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                legal_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                legal_chunks = text_splitter.split_documents(legal_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(legal_chunks)
                    self.fraud_databases['legal_precedents'] = legal_chunks
                    
        except Exception as e:
            logger.error(f"Error loading legal precedents: {e}")
    
    def load_investigation_guides(self):
        """Load fraud investigation guides"""
        try:
            investigation_path = self.config.get("investigation_guides_path", "./data/investigation_guides")
            if os.path.exists(investigation_path):
                loader = DirectoryLoader(
                    investigation_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                investigation_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                investigation_chunks = text_splitter.split_documents(investigation_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(investigation_chunks)
                    self.fraud_databases['investigation_guides'] = investigation_chunks
                    
        except Exception as e:
            logger.error(f"Error loading investigation guides: {e}")
    
    def load_risk_frameworks(self):
        """Load risk assessment frameworks"""
        try:
            risk_path = self.config.get("risk_frameworks_path", "./data/risk_frameworks")
            if os.path.exists(risk_path):
                loader = DirectoryLoader(
                    risk_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                risk_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                risk_chunks = text_splitter.split_documents(risk_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(risk_chunks)
                    self.fraud_databases['risk_frameworks'] = risk_chunks
                    
        except Exception as e:
            logger.error(f"Error loading risk frameworks: {e}")
    
    def retrieve_fraud_knowledge(self, query: str, claim_context: Optional[Dict[str, Any]] = None, use_ensemble: bool = True) -> List[str]:
        """Retrieve relevant fraud detection knowledge with enhanced context awareness"""
        try:
            # Check cache first
            cache_key = f"{query}_{hash(str(claim_context))}"
            if cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            # Enhance query with claim context
            enhanced_query = self.enhance_query_with_claim_context(query, claim_context)
            
            if use_ensemble and hasattr(self, 'ensemble_retriever'):
                docs = self.ensemble_retriever.get_relevant_documents(enhanced_query)
            else:
                docs = self.retriever.get_relevant_documents(enhanced_query)
            
            # Filter and rank results
            filtered_docs = self.filter_relevant_documents(docs, query, claim_context)
            
            # Cache results
            self.retrieval_cache[cache_key] = [doc.page_content for doc in filtered_docs]
            
            return [doc.page_content for doc in filtered_docs]
            
        except Exception as e:
            logger.error(f"Error retrieving fraud knowledge: {e}")
            return []
    
    def enhance_query_with_claim_context(self, query: str, claim_context: Optional[Dict[str, Any]] = None) -> str:
        """Enhance query with claim context"""
        try:
            if not claim_context:
                return query
            
            enhanced_parts = [query]
            
            # Add claim amount information
            if 'billed_amount' in claim_context:
                enhanced_parts.append(f"Billed Amount: ${claim_context['billed_amount']}")
            
            # Add provider information
            if 'provider_type' in claim_context:
                enhanced_parts.append(f"Provider Type: {claim_context['provider_type']}")
            
            # Add diagnosis information
            if 'diagnosis_codes' in claim_context:
                enhanced_parts.append(f"Diagnosis Codes: {', '.join(claim_context['diagnosis_codes'])}")
            
            # Add procedure information
            if 'procedure_codes' in claim_context:
                enhanced_parts.append(f"Procedure Codes: {', '.join(claim_context['procedure_codes'])}")
            
            # Add temporal information
            if 'claim_date' in claim_context and 'service_date' in claim_context:
                days_diff = (claim_context['claim_date'] - claim_context['service_date']).days
                enhanced_parts.append(f"Days between service and claim: {days_diff}")
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query with claim context: {e}")
            return query
    
    def filter_relevant_documents(self, docs: List, query: str, claim_context: Optional[Dict[str, Any]] = None) -> List:
        """Filter and rank documents based on relevance"""
        try:
            if not docs:
                return []
            
            # Calculate relevance scores
            scored_docs = []
            for doc in docs:
                relevance_score = self.calculate_fraud_relevance_score(doc.page_content, query, claim_context)
                scored_docs.append((doc, relevance_score))
            
            # Sort by relevance score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top documents
            return [doc for doc, score in scored_docs[:5]]
            
        except Exception as e:
            logger.error(f"Error filtering documents: {e}")
            return docs[:5]
    
    def calculate_fraud_relevance_score(self, content: str, query: str, claim_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate relevance score for fraud detection content"""
        try:
            score = 0.0
            
            # Basic text similarity
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if query_words:
                word_overlap = len(query_words.intersection(content_words)) / len(query_words)
                score += word_overlap * 0.4
            
            # Context relevance
            if claim_context:
                if 'provider_type' in claim_context and claim_context['provider_type'].lower() in content.lower():
                    score += 0.2
                
                if 'billed_amount' in claim_context:
                    amount = claim_context['billed_amount']
                    if amount > 10000 and 'high value' in content.lower():
                        score += 0.2
                    elif amount < 1000 and 'low value' in content.lower():
                        score += 0.1
                
                if 'diagnosis_codes' in claim_context:
                    for code in claim_context['diagnosis_codes']:
                        if code.lower() in content.lower():
                            score += 0.1
            
            # Content type preference
            if any(keyword in content.lower() for keyword in ['fraud', 'suspicious', 'anomaly', 'pattern']):
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating fraud relevance score: {e}")
            return 0.5
    
    def get_fraud_context(self, query: str, claim_context: Optional[Dict[str, Any]] = None) -> str:
        """Get comprehensive fraud detection context"""
        try:
            knowledge = self.retrieve_fraud_knowledge(query, claim_context)
            
            if not knowledge:
                return "No specific fraud detection information found."
            
            # Combine knowledge into context
            context_parts = []
            for i, info in enumerate(knowledge[:3], 1):
                context_parts.append(f"Fraud Detection Information {i}: {info}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting fraud context: {e}")
            return "Unable to retrieve fraud detection context."
    
    def extract_fraud_entities(self, query: str) -> Dict[str, Any]:
        """Extract fraud-related entities from query"""
        try:
            entities = {
                'fraud_types': [],
                'risk_indicators': [],
                'regulatory_mentions': [],
                'investigation_actions': []
            }
            
            query_lower = query.lower()
            
            # Extract fraud types
            fraud_keywords = ['billing fraud', 'upcoding', 'unbundling', 'phantom billing', 'kickback']
            for keyword in fraud_keywords:
                if keyword in query_lower:
                    entities['fraud_types'].append(keyword)
            
            # Extract risk indicators
            risk_keywords = ['suspicious', 'anomaly', 'unusual', 'excessive', 'inconsistent']
            for keyword in risk_keywords:
                if keyword in query_lower:
                    entities['risk_indicators'].append(keyword)
            
            # Extract regulatory mentions
            regulatory_keywords = ['compliance', 'regulation', 'law', 'policy', 'guideline']
            for keyword in regulatory_keywords:
                if keyword in query_lower:
                    entities['regulatory_mentions'].append(keyword)
            
            # Extract investigation actions
            action_keywords = ['investigate', 'audit', 'review', 'examine', 'analyze']
            for keyword in action_keywords:
                if keyword in query_lower:
                    entities['investigation_actions'].append(keyword)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting fraud entities: {e}")
            return {}
    
    def generate_fraud_analysis(self, claim_data: InsuranceClaim, fraud_indicators: List[str]) -> Dict[str, str]:
        """Generate enhanced fraud analysis using RAG"""
        try:
            # Create claim context
            claim_context = {
                'billed_amount': claim_data.billed_amount,
                'provider_type': claim_data.provider_type,
                'diagnosis_codes': claim_data.diagnosis_codes,
                'procedure_codes': claim_data.procedure_codes,
                'claim_date': claim_data.claim_date,
                'service_date': claim_data.service_date
            }
            
            # Create query for fraud analysis
            query = f"""
            Insurance claim analysis for fraud detection:
            Claim ID: {claim_data.claim_id}
            Provider: {claim_data.provider_id}
            Amount: ${claim_data.billed_amount}
            Diagnosis: {claim_data.diagnosis_codes}
            Procedures: {claim_data.procedure_codes}
            Fraud indicators: {fraud_indicators}
            
            Provide detailed analysis and recommendations.
            """
            
            # Retrieve relevant knowledge
            knowledge = self.retrieve_fraud_knowledge(query, claim_context)
            
            # Generate analysis using LLM
            prompt_template = PromptTemplate(
                input_variables=["claim_info", "fraud_indicators", "knowledge", "claim_context"],
                template="""
                Based on the following insurance claim information, fraud detection knowledge, and claim context, 
                provide a comprehensive fraud analysis:
                
                Claim Information: {claim_info}
                Claim Context: {claim_context}
                Fraud Indicators: {fraud_indicators}
                Knowledge Base: {knowledge}
                
                Provide:
                1. Risk Assessment
                2. Suspicious Patterns Identified
                3. Regulatory Compliance Analysis
                4. Recommended Actions
                5. Investigation Priority
                6. Evidence Requirements
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = chain.run({
                "claim_info": str(claim_data.__dict__),
                "claim_context": str(claim_context),
                "fraud_indicators": str(fraud_indicators),
                "knowledge": "\n".join(knowledge)
            })
            
            # Parse response into sections
            sections = response.split('\n\n')
            analysis = {}
            
            for section in sections:
                if 'risk' in section.lower():
                    analysis['risk_assessment'] = section.strip()
                elif 'pattern' in section.lower():
                    analysis['suspicious_patterns'] = section.strip()
                elif 'compliance' in section.lower():
                    analysis['regulatory_compliance'] = section.strip()
                elif 'recommend' in section.lower():
                    analysis['recommended_actions'] = section.strip()
                elif 'investigation' in section.lower():
                    analysis['investigation_priority'] = section.strip()
                elif 'evidence' in section.lower():
                    analysis['evidence_requirements'] = section.strip()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating fraud analysis: {e}")
            return {
                'risk_assessment': 'Standard risk assessment required',
                'suspicious_patterns': 'No specific patterns identified',
                'regulatory_compliance': 'Compliance review needed',
                'recommended_actions': 'Review claim manually',
                'investigation_priority': 'Medium priority',
                'evidence_requirements': 'Standard documentation required'
            }

class FraudDetectionDataset(Dataset):
    """Custom PyTorch dataset for fraud detection"""
    
    def __init__(self, claims: List[InsuranceClaim], labels: List[int], transform=None):
        self.claims = claims
        self.labels = labels
        self.transform = transform
        self.feature_extractor = FraudFeatureExtractor()
        self.processed_data = self.preprocess_data()
    
    def preprocess_data(self) -> np.ndarray:
        """Preprocess claim data for fraud detection"""
        processed_data = []
        
        for claim in self.claims:
            features = self.feature_extractor.extract_features(claim)
            processed_data.append(features)
        
        return np.array(processed_data)
    
    def __len__(self):
        return len(self.claims)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.processed_data[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label

class FraudFeatureExtractor:
    """Extract features from insurance claims for fraud detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def extract_features(self, claim: InsuranceClaim) -> List[float]:
        """Extract numerical features from claim"""
        features = []
        
        # Basic claim features
        features.extend([
            claim.billed_amount,
            claim.allowed_amount,
            claim.paid_amount,
            (claim.billed_amount - claim.allowed_amount) / claim.billed_amount if claim.billed_amount > 0 else 0,
            (claim.allowed_amount - claim.paid_amount) / claim.allowed_amount if claim.allowed_amount > 0 else 0
        ])
        
        # Temporal features
        claim_date = claim.claim_date
        service_date = claim.service_date
        days_diff = (claim_date - service_date).days
        features.append(days_diff)
        
        # Diagnosis and procedure features
        features.extend([
            len(claim.diagnosis_codes),
            len(claim.procedure_codes),
            sum(len(code) for code in claim.diagnosis_codes),
            sum(len(code) for code in claim.procedure_codes)
        ])
        
        # Patient demographics features
        demographics = claim.patient_demographics
        features.extend([
            demographics.get('age', 0),
            1 if demographics.get('gender', '').lower() == 'male' else 0,
            demographics.get('income_level', 0),
            demographics.get('previous_claims', 0)
        ])
        
        # Provider features
        provider_info = claim.provider_info
        features.extend([
            provider_info.get('years_in_practice', 0),
            provider_info.get('total_claims', 0),
            provider_info.get('average_claim_amount', 0),
            provider_info.get('fraud_history_score', 0)
        ])
        
        # Claim type encoding
        claim_type_encoded = self.encode_categorical(claim.claim_type, 'claim_type')
        features.append(claim_type_encoded)
        
        # Service type encoding
        service_type_encoded = self.encode_categorical(claim.service_type, 'service_type')
        features.append(service_type_encoded)
        
        # Provider type encoding
        provider_type_encoded = self.encode_categorical(claim.provider_type, 'provider_type')
        features.append(provider_type_encoded)
        
        return features
    
    def encode_categorical(self, value: str, category: str) -> float:
        """Encode categorical values"""
        if category not in self.label_encoders:
            self.label_encoders[category] = LabelEncoder()
            # This would be fitted with training data in practice
            return 0.0
        
        try:
            return float(self.label_encoders[category].transform([value])[0])
        except:
            return 0.0

class FraudDetectionNeuralNetwork(nn.Module):
    """Neural network for fraud detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        super(FraudDetectionNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

class FraudDetectionSystem:
    """Main fraud detection system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = FraudDetectionRAGSystem(config)
        self.models = {}
        self.feature_extractor = FraudFeatureExtractor()
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all fraud detection models"""
        try:
            # Initialize traditional ML models
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            self.models['logistic_regression'] = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            self.models['svm'] = SVC(
                probability=True,
                random_state=42
            )
            
            # Initialize neural network
            input_dim = self.config.get('input_dim', 20)  # Number of features
            hidden_dims = self.config.get('hidden_dims', [64, 32, 16])
            
            self.models['neural_network'] = FraudDetectionNeuralNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            # Load pre-trained models if available
            self.load_model_weights()
            
            logger.info("Fraud detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def load_model_weights(self):
        """Load pre-trained model weights"""
        try:
            # Load traditional ML models
            for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm']:
                model_path = self.config.get(f"{model_name}_model_path")
                if model_path and os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"{model_name} model loaded")
            
            # Load neural network weights
            nn_path = self.config.get("neural_network_weights_path")
            if nn_path and os.path.exists(nn_path):
                self.models['neural_network'].load_state_dict(
                    torch.load(nn_path, map_location=self.device)
                )
                logger.info("Neural network weights loaded")
                
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
    
    def extract_claim_features(self, claim: InsuranceClaim) -> np.ndarray:
        """Extract features from claim"""
        try:
            features = self.feature_extractor.extract_features(claim)
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features = self.scaler.fit_transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting claim features: {e}")
            raise
    
    def detect_fraud_patterns(self, claim: InsuranceClaim) -> List[str]:
        """Detect fraud patterns in claim"""
        try:
            patterns = []
            
            # Check for billing anomalies
            if claim.billed_amount > claim.allowed_amount * 1.5:
                patterns.append("Excessive billing amount")
            
            if claim.billed_amount > 10000:  # High-value claim
                patterns.append("High-value claim")
            
            # Check for temporal anomalies
            days_diff = (claim.claim_date - claim.service_date).days
            if days_diff > 90:
                patterns.append("Delayed claim submission")
            
            # Check for diagnosis-procedure mismatches
            if len(claim.diagnosis_codes) != len(claim.procedure_codes):
                patterns.append("Diagnosis-procedure count mismatch")
            
            # Check for provider patterns
            provider_info = claim.provider_info
            if provider_info.get('fraud_history_score', 0) > 0.7:
                patterns.append("Provider with fraud history")
            
            if provider_info.get('average_claim_amount', 0) > 5000:
                patterns.append("Provider with high average claims")
            
            # Check for patient patterns
            demographics = claim.patient_demographics
            if demographics.get('previous_claims', 0) > 10:
                patterns.append("Patient with many previous claims")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting fraud patterns: {e}")
            return []
    
    def predict_fraud(self, claim: InsuranceClaim, model_name: str = 'ensemble') -> Tuple[float, float]:
        """Predict fraud probability using specified model"""
        try:
            # Extract features
            features = self.extract_claim_features(claim)
            
            if model_name == 'ensemble':
                # Use ensemble of models
                predictions = []
                weights = [0.3, 0.25, 0.25, 0.2]  # Weights for different models
                
                for i, (model_key, weight) in enumerate(zip(['random_forest', 'gradient_boosting', 'logistic_regression', 'svm'], weights)):
                    if model_key in self.models:
                        model = self.models[model_key]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features)[0][1]
                        else:
                            proba = model.predict(features)[0]
                        predictions.append(proba * weight)
                
                fraud_probability = sum(predictions)
                confidence_score = 0.8  # Ensemble confidence
                
            elif model_name == 'neural_network':
                # Use neural network
                model = self.models['neural_network']
                model.eval()
                
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).to(self.device)
                    output = model(features_tensor)
                    fraud_probability = output.item()
                    confidence_score = 0.85
                    
            else:
                # Use specific model
                if model_name in self.models:
                    model = self.models[model_name]
                    if hasattr(model, 'predict_proba'):
                        fraud_probability = model.predict_proba(features)[0][1]
                    else:
                        fraud_probability = model.predict(features)[0]
                    confidence_score = 0.75
                else:
                    raise ValueError(f"Unknown model: {model_name}")
            
            return fraud_probability, confidence_score
            
        except Exception as e:
            logger.error(f"Error predicting fraud: {e}")
            raise
    
    def analyze_claim(self, claim: InsuranceClaim) -> FraudDetectionResult:
        """Analyze claim for fraud"""
        try:
            # Detect fraud patterns
            fraud_patterns = self.detect_fraud_patterns(claim)
            
            # Predict fraud probability
            fraud_probability, confidence_score = self.predict_fraud(claim)
            
            # Calculate fraud score (normalized)
            fraud_score = fraud_probability * 100
            
            # Determine risk level
            if fraud_score >= 80:
                risk_level = "High"
            elif fraud_score >= 50:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Generate fraud indicators
            fraud_indicators = self.generate_fraud_indicators(claim, fraud_patterns)
            
            # Generate suspicious patterns
            suspicious_patterns = self.generate_suspicious_patterns(claim, fraud_patterns)
            
            # Generate recommendation
            recommendation = self.generate_recommendation(fraud_score, risk_level, fraud_patterns)
            
            # Generate detailed analysis using RAG
            fraud_analysis = self.rag_system.generate_fraud_analysis(claim, fraud_indicators)
            
            return FraudDetectionResult(
                claim_id=claim.claim_id,
                fraud_probability=fraud_probability,
                fraud_score=fraud_score,
                risk_level=risk_level,
                fraud_indicators=fraud_indicators,
                suspicious_patterns=suspicious_patterns,
                recommendation=recommendation,
                confidence_score=confidence_score,
                model_used="ensemble",
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing claim: {e}")
            raise
    
    def generate_fraud_indicators(self, claim: InsuranceClaim, patterns: List[str]) -> List[str]:
        """Generate fraud indicators"""
        try:
            indicators = []
            
            # Add detected patterns
            indicators.extend(patterns)
            
            # Add additional indicators based on claim characteristics
            if claim.billed_amount > claim.allowed_amount * 2:
                indicators.append("Billed amount significantly higher than allowed")
            
            if len(claim.diagnosis_codes) > 5:
                indicators.append("Unusually high number of diagnoses")
            
            if len(claim.procedure_codes) > 3:
                indicators.append("Unusually high number of procedures")
            
            # Provider-based indicators
            provider_info = claim.provider_info
            if provider_info.get('years_in_practice', 0) < 2:
                indicators.append("New provider")
            
            if provider_info.get('total_claims', 0) > 1000:
                indicators.append("Provider with very high claim volume")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error generating fraud indicators: {e}")
            return []
    
    def generate_suspicious_patterns(self, claim: InsuranceClaim, patterns: List[str]) -> List[str]:
        """Generate suspicious patterns"""
        try:
            suspicious_patterns = []
            
            # Add detected patterns
            suspicious_patterns.extend(patterns)
            
            # Add pattern analysis
            if claim.billed_amount > 5000:
                suspicious_patterns.append("High-value claim pattern")
            
            if (claim.claim_date - claim.service_date).days > 30:
                suspicious_patterns.append("Delayed submission pattern")
            
            # Check for weekend/holiday patterns
            if claim.service_date.weekday() >= 5:  # Weekend
                suspicious_patterns.append("Weekend service pattern")
            
            return suspicious_patterns
            
        except Exception as e:
            logger.error(f"Error generating suspicious patterns: {e}")
            return []
    
    def generate_recommendation(self, fraud_score: float, risk_level: str, patterns: List[str]) -> str:
        """Generate recommendation based on analysis"""
        try:
            if risk_level == "High":
                if len(patterns) > 3:
                    return "Immediate investigation required - multiple fraud indicators detected"
                else:
                    return "High priority investigation recommended"
            elif risk_level == "Medium":
                if len(patterns) > 2:
                    return "Investigation recommended - several suspicious patterns"
                else:
                    return "Manual review recommended"
            else:
                if len(patterns) > 0:
                    return "Standard review - minor concerns detected"
                else:
                    return "No immediate action required"
                    
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "Manual review recommended"

# Pydantic models for API
class InsuranceClaimRequest(BaseModel):
    claim_id: str
    patient_id: str
    provider_id: str
    claim_date: datetime
    service_date: datetime
    diagnosis_codes: List[str]
    procedure_codes: List[str]
    billed_amount: float = Field(..., gt=0)
    allowed_amount: float = Field(..., gt=0)
    paid_amount: float = Field(..., gt=0)
    provider_type: str
    service_type: str
    claim_type: str
    patient_demographics: Dict[str, Any] = {}
    provider_info: Dict[str, Any] = {}
    claim_details: Dict[str, Any] = {}

class FraudDetectionResponse(BaseModel):
    claim_id: str
    fraud_probability: float
    fraud_score: float
    risk_level: str
    fraud_indicators: List[str]
    suspicious_patterns: List[str]
    recommendation: str
    confidence_score: float
    model_used: str
    analysis_timestamp: datetime

class BatchFraudDetectionRequest(BaseModel):
    claims: List[InsuranceClaimRequest]
    model_name: str = Field(default="ensemble", regex="^(ensemble|random_forest|gradient_boosting|logistic_regression|svm|neural_network)$")

class BatchFraudDetectionResponse(BaseModel):
    results: List[FraudDetectionResponse]
    total_claims: int
    high_risk_claims: int
    average_fraud_score: float
    timestamp: datetime

# Initialize the system
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
    "chroma_db_path": "./fraud_detection_chroma_db",
    "llm_model_name": "gpt-4",
    "input_dim": 20,
    "hidden_dims": [64, 32, 16],
    "random_forest_model_path": "./models/fraud_random_forest.pkl",
    "gradient_boosting_model_path": "./models/fraud_gradient_boosting.pkl",
    "logistic_regression_model_path": "./models/fraud_logistic_regression.pkl",
    "svm_model_path": "./models/fraud_svm.pkl",
    "neural_network_weights_path": "./models/fraud_neural_network.pth",
    "fraud_patterns_path": "./data/fraud_patterns",
    "regulations_path": "./data/regulations",
    "case_studies_path": "./data/case_studies"
}

fraud_detection_system = FraudDetectionSystem(config)

@app.post("/detect_fraud", response_model=FraudDetectionResponse)
async def detect_fraud(request: InsuranceClaimRequest):
    """Detect fraud in insurance claim"""
    try:
        # Convert request to InsuranceClaim
        claim = InsuranceClaim(
            claim_id=request.claim_id,
            patient_id=request.patient_id,
            provider_id=request.provider_id,
            claim_date=request.claim_date,
            service_date=request.service_date,
            diagnosis_codes=request.diagnosis_codes,
            procedure_codes=request.procedure_codes,
            billed_amount=request.billed_amount,
            allowed_amount=request.allowed_amount,
            paid_amount=request.paid_amount,
            provider_type=request.provider_type,
            service_type=request.service_type,
            claim_type=request.claim_type,
            patient_demographics=request.patient_demographics,
            provider_info=request.provider_info,
            claim_details=request.claim_details
        )
        
        # Analyze claim for fraud
        result = fraud_detection_system.analyze_claim(claim)
        
        # Convert to response format
        response = FraudDetectionResponse(
            claim_id=result.claim_id,
            fraud_probability=result.fraud_probability,
            fraud_score=result.fraud_score,
            risk_level=result.risk_level,
            fraud_indicators=result.fraud_indicators,
            suspicious_patterns=result.suspicious_patterns,
            recommendation=result.recommendation,
            confidence_score=result.confidence_score,
            model_used=result.model_used,
            analysis_timestamp=result.analysis_timestamp
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in detect fraud endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_detect_fraud", response_model=BatchFraudDetectionResponse)
async def batch_detect_fraud(request: BatchFraudDetectionRequest):
    """Detect fraud in multiple insurance claims"""
    try:
        results = []
        total_fraud_score = 0.0
        high_risk_count = 0
        
        # Process each claim
        for claim_request in request.claims:
            # Convert to InsuranceClaim
            claim = InsuranceClaim(
                claim_id=claim_request.claim_id,
                patient_id=claim_request.patient_id,
                provider_id=claim_request.provider_id,
                claim_date=claim_request.claim_date,
                service_date=claim_request.service_date,
                diagnosis_codes=claim_request.diagnosis_codes,
                procedure_codes=claim_request.procedure_codes,
                billed_amount=claim_request.billed_amount,
                allowed_amount=claim_request.allowed_amount,
                paid_amount=claim_request.paid_amount,
                provider_type=claim_request.provider_type,
                service_type=claim_request.service_type,
                claim_type=claim_request.claim_type,
                patient_demographics=claim_request.patient_demographics,
                provider_info=claim_request.provider_info,
                claim_details=claim_request.claim_details
            )
            
            # Analyze claim for fraud
            result = fraud_detection_system.analyze_claim(claim)
            
            # Convert to response format
            response = FraudDetectionResponse(
                claim_id=result.claim_id,
                fraud_probability=result.fraud_probability,
                fraud_score=result.fraud_score,
                risk_level=result.risk_level,
                fraud_indicators=result.fraud_indicators,
                suspicious_patterns=result.suspicious_patterns,
                recommendation=result.recommendation,
                confidence_score=result.confidence_score,
                model_used=result.model_used,
                analysis_timestamp=result.analysis_timestamp
            )
            
            results.append(response)
            total_fraud_score += result.fraud_score
            
            # Count high-risk claims
            if result.risk_level == "High":
                high_risk_count += 1
        
        # Calculate average fraud score
        average_fraud_score = total_fraud_score / len(results) if results else 0.0
        
        return BatchFraudDetectionResponse(
            results=results,
            total_claims=len(results),
            high_risk_claims=high_risk_count,
            average_fraud_score=average_fraud_score,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in batch detect fraud endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/list")
async def get_models_list():
    """Get list of available fraud detection models"""
    try:
        return {
            "models": list(fraud_detection_system.models.keys()),
            "model_descriptions": {
                "ensemble": "Combination of multiple models for best performance",
                "random_forest": "Random Forest classifier for fraud detection",
                "gradient_boosting": "Gradient Boosting classifier",
                "logistic_regression": "Logistic Regression classifier",
                "svm": "Support Vector Machine classifier",
                "neural_network": "Deep neural network for fraud detection"
            },
            "recommended_model": "ensemble"
        }
    except Exception as e:
        logger.error(f"Error getting models list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "Health Insurance Fraud Detection API",
        "models_loaded": len(fraud_detection_system.models),
        "rag_system_ready": fraud_detection_system.rag_system.retriever is not None,
        "device": str(fraud_detection_system.device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006) 