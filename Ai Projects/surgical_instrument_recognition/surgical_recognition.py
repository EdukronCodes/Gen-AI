"""
CNN-Based Surgical Instrument Recognition System with Enhanced RAG Integration
A comprehensive system for recognizing and classifying surgical instruments using deep learning and surgical knowledge retrieval
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import fastapi
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

# Medical and Surgical Libraries
import spacy
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

import chromadb
from chromadb.config import Settings
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import io
import base64
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
    SURGICAL_INSTRUMENT_CLASSES,
    RECOGNITION_PARAMS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Surgical Instrument Recognition API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class SurgicalInstrument:
    """Surgical instrument data structure with enhanced metadata"""
    instrument_id: str
    name: str
    category: str
    subcategory: str
    manufacturer: str
    material: str
    size: str
    sterilization_requirements: str
    usage_instructions: str
    safety_guidelines: str
    maintenance_procedures: str
    surgical_context: Dict[str, Any]
    research_evidence: Dict[str, Any]

@dataclass
class RecognitionResult:
    """Result of surgical instrument recognition with enhanced RAG insights"""
    instrument_id: str
    instrument_name: str
    category: str
    confidence_score: float
    bounding_box: List[int]
    usage_instructions: str
    safety_guidelines: str
    maintenance_info: str
    similar_instruments: List[str]
    sterilization_status: str
    surgical_context: Dict[str, Any]
    research_insights: Dict[str, Any]

class SurgicalRAGSystem:
    """Enhanced RAG system for surgical knowledge retrieval with instrument context awareness"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.bm25_retriever = None
        self.knowledge_base = {}
        self.retrieval_cache = {}
        self.surgical_databases = {}
        self.initialize_vector_store()
        self.load_surgical_sources()
        self.setup_retrievers()
    
    def initialize_vector_store(self):
        """Initialize vector store with enhanced surgical knowledge"""
        try:
            # Initialize ChromaDB with surgical-specific settings
            self.vector_store = ChromaDB(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                client_settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("Surgical vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def setup_retrievers(self):
        """Setup multiple retrieval strategies for enhanced surgical information retrieval"""
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
            
            logger.info("Surgical retrievers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
    
    def load_surgical_sources(self):
        """Load comprehensive surgical knowledge sources from various repositories"""
        try:
            # Load from local surgical knowledge base
            if os.path.exists(KNOWLEDGE_SOURCES.get("local", "")):
                self.load_surgical_guidelines()
            
            # Load from PubMed surgical articles
            self.load_pubmed_surgical_articles()
            
            # Load from surgical instrument catalogs
            self.load_instrument_catalogs()
            
            # Load from surgical procedure guides
            self.load_procedure_guides()
            
            # Load from safety and sterilization guidelines
            self.load_safety_guidelines()
            
            # Load from surgical training materials
            self.load_training_materials()
            
            logger.info(f"Loaded {len(self.knowledge_base)} surgical knowledge sources")
            
        except Exception as e:
            logger.error(f"Error loading surgical sources: {e}")
    
    def load_surgical_guidelines(self):
        """Load surgical practice guidelines and standards"""
        try:
            # Load from surgical guidelines databases
            guidelines = [
                {
                    "title": "Surgical Instrument Sterilization Guidelines",
                    "content": "All surgical instruments must undergo proper sterilization...",
                    "source": "Association of periOperative Registered Nurses",
                    "year": 2024,
                    "category": "sterilization"
                },
                {
                    "title": "Surgical Safety Checklist Implementation",
                    "content": "The surgical safety checklist includes instrument verification...",
                    "source": "World Health Organization",
                    "year": 2024,
                    "category": "safety"
                },
                {
                    "title": "Minimally Invasive Surgery Instrument Standards",
                    "content": "Laparoscopic instruments require specific handling procedures...",
                    "source": "Society of American Gastrointestinal and Endoscopic Surgeons",
                    "year": 2024,
                    "category": "laparoscopic"
                }
            ]
            
            for guideline in guidelines:
                self.knowledge_base[f"guideline_{guideline['source']}_{guideline['title']}"] = {
                    "content": guideline['content'],
                    "metadata": guideline,
                    "type": "guideline"
                }
                
        except Exception as e:
            logger.error(f"Error loading surgical guidelines: {e}")
    
    def load_pubmed_surgical_articles(self):
        """Load surgical research articles from PubMed"""
        try:
            # This would integrate with PubMed API
            # For now, we'll simulate with sample data
            surgical_articles = [
                {
                    "title": "Surgical instrument recognition using computer vision",
                    "abstract": "This study examines the effectiveness of AI-based surgical instrument recognition...",
                    "authors": ["Dr. Chen", "Dr. Wang"],
                    "journal": "Journal of Surgical Research",
                    "year": 2024,
                    "pmid": "12345678"
                },
                {
                    "title": "Sterilization protocols for robotic surgical instruments",
                    "abstract": "Robotic surgical instruments require specialized sterilization procedures...",
                    "authors": ["Dr. Johnson", "Dr. Smith"],
                    "journal": "Surgical Endoscopy",
                    "year": 2024,
                    "pmid": "87654321"
                }
            ]
            
            for article in surgical_articles:
                self.knowledge_base[f"pubmed_{article['pmid']}"] = {
                    "content": f"{article['title']}\n{article['abstract']}",
                    "metadata": article,
                    "type": "pubmed"
                }
                
        except Exception as e:
            logger.error(f"Error loading PubMed surgical articles: {e}")
    
    def load_instrument_catalogs(self):
        """Load surgical instrument catalogs and specifications"""
        try:
            # Load instrument catalogs
            catalogs = [
                {
                    "instrument": "Scalpel",
                    "specifications": "Disposable blade with handle, various blade sizes available",
                    "usage": "Cutting tissue during surgical procedures",
                    "safety": "Handle with care, dispose of blades properly",
                    "sterilization": "Autoclave sterilization required for reusable handles"
                },
                {
                    "instrument": "Forceps",
                    "specifications": "Tissue grasping instrument with serrated tips",
                    "usage": "Grasping and holding tissue during procedures",
                    "safety": "Avoid excessive force to prevent tissue damage",
                    "sterilization": "Standard autoclave sterilization"
                },
                {
                    "instrument": "Laparoscopic Trocar",
                    "specifications": "Hollow tube for laparoscopic access",
                    "usage": "Creating access ports for laparoscopic surgery",
                    "safety": "Insert carefully to avoid organ injury",
                    "sterilization": "High-level disinfection or sterilization required"
                }
            ]
            
            for catalog in catalogs:
                self.knowledge_base[f"catalog_{catalog['instrument']}"] = {
                    "content": f"Instrument: {catalog['instrument']}\nSpecifications: {catalog['specifications']}\nUsage: {catalog['usage']}",
                    "metadata": catalog,
                    "type": "catalog"
                }
                
        except Exception as e:
            logger.error(f"Error loading instrument catalogs: {e}")
    
    def load_procedure_guides(self):
        """Load surgical procedure guides and protocols"""
        try:
            # Load procedure guides
            procedures = [
                {
                    "procedure": "Laparoscopic Cholecystectomy",
                    "instruments": ["Laparoscope", "Graspers", "Scissors", "Clip applier"],
                    "setup": "Patient positioned supine, trocars placed in standard positions",
                    "safety": "Identify critical structures before dissection"
                },
                {
                    "procedure": "Open Appendectomy",
                    "instruments": ["Scalpel", "Forceps", "Retractors", "Sutures"],
                    "setup": "Patient positioned supine, right lower quadrant incision",
                    "safety": "Ensure proper identification of appendix"
                }
            ]
            
            for procedure in procedures:
                self.knowledge_base[f"procedure_{procedure['procedure']}"] = {
                    "content": f"Procedure: {procedure['procedure']}\nInstruments: {', '.join(procedure['instruments'])}\nSetup: {procedure['setup']}",
                    "metadata": procedure,
                    "type": "procedure"
                }
                
        except Exception as e:
            logger.error(f"Error loading procedure guides: {e}")
    
    def load_safety_guidelines(self):
        """Load surgical safety and sterilization guidelines"""
        try:
            # Load safety guidelines
            safety_guidelines = [
                {
                    "topic": "Instrument Sterilization",
                    "guidelines": "All reusable instruments must be sterilized according to manufacturer instructions",
                    "compliance": "Required for all surgical procedures",
                    "monitoring": "Regular sterilization validation required"
                },
                {
                    "topic": "Sharps Safety",
                    "guidelines": "Sharps must be handled with extreme care and disposed of properly",
                    "compliance": "Mandatory safety protocol",
                    "monitoring": "Incident reporting required"
                }
            ]
            
            for guideline in safety_guidelines:
                self.knowledge_base[f"safety_{guideline['topic']}"] = {
                    "content": f"Topic: {guideline['topic']}\nGuidelines: {guideline['guidelines']}\nCompliance: {guideline['compliance']}",
                    "metadata": guideline,
                    "type": "safety"
                }
                
        except Exception as e:
            logger.error(f"Error loading safety guidelines: {e}")
    
    def load_training_materials(self):
        """Load surgical training and educational materials"""
        try:
            # Load training materials
            training_materials = [
                {
                    "topic": "Surgical Instrument Handling",
                    "content": "Proper handling techniques for surgical instruments",
                    "learning_objectives": ["Identify instruments", "Understand usage", "Practice safety"],
                    "target_audience": "Surgical residents and nurses"
                }
            ]
            
            for material in training_materials:
                self.knowledge_base[f"training_{material['topic']}"] = {
                    "content": f"Topic: {material['topic']}\nContent: {material['content']}",
                    "metadata": material,
                    "type": "training"
                }
                
        except Exception as e:
            logger.error(f"Error loading training materials: {e}")
    
    def retrieve_surgical_knowledge(self, query: str, instrument_context: Dict[str, Any] = None, 
                                   top_k: int = 5, use_ensemble: bool = True) -> List[Dict]:
        """Enhanced retrieval with instrument context awareness and multiple strategies"""
        try:
            # Check cache first
            cache_key = f"{query}_{str(instrument_context)}_{top_k}_{use_ensemble}"
            if cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            # Enhance query with instrument context
            if instrument_context:
                enhanced_query = self.enhance_query_with_instrument_context(query, instrument_context)
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
                    "relevance_score": self.calculate_surgical_relevance_score(enhanced_query, doc.page_content, instrument_context),
                    "evidence_level": doc.metadata.get("evidence_level", "unknown")
                }
                retrieved_info.append(info)
            
            # Sort by relevance score
            retrieved_info.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Cache results
            self.retrieval_cache[cache_key] = retrieved_info[:top_k]
            
            logger.info(f"Retrieved {len(retrieved_info)} relevant documents for surgical query: {query[:50]}...")
            return retrieved_info[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving surgical knowledge: {e}")
            return []
    
    def enhance_query_with_instrument_context(self, query: str, instrument_context: Dict[str, Any]) -> str:
        """Enhance query with instrument-specific context"""
        try:
            enhanced_parts = [query]
            
            # Add instrument category context
            if 'category' in instrument_context:
                enhanced_parts.append(f"category {instrument_context['category']}")
            if 'subcategory' in instrument_context:
                enhanced_parts.append(f"subcategory {instrument_context['subcategory']}")
            
            # Add surgical procedure context
            if 'surgical_procedure' in instrument_context:
                procedure = instrument_context['surgical_procedure']
                enhanced_parts.append(f"procedure {procedure}")
            
            # Add sterilization context
            if 'sterilization_requirements' in instrument_context:
                sterilization = instrument_context['sterilization_requirements']
                enhanced_parts.append(f"sterilization {sterilization}")
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query with instrument context: {e}")
            return query
    
    def calculate_surgical_relevance_score(self, query: str, content: str, instrument_context: Dict[str, Any] = None) -> float:
        """Calculate relevance score for surgical content with instrument context"""
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
            
            # Boost score for surgical-related terms
            surgical_terms = [
                "surgical", "instrument", "procedure", "operation", "sterilization",
                "safety", "guideline", "protocol", "training", "catalog",
                "usage", "maintenance", "specification", "manufacturer"
            ]
            surgical_boost = sum(1 for term in surgical_terms if term in content.lower())
            relevance += surgical_boost * 0.1
            
            # Boost for instrument-specific context
            if instrument_context:
                # Boost for category-specific content
                if 'category' in instrument_context:
                    category = instrument_context['category']
                    if category.lower() in content.lower():
                        relevance += 0.15
                
                # Boost for procedure-specific content
                if 'surgical_procedure' in instrument_context:
                    procedure = instrument_context['surgical_procedure']
                    if procedure.lower() in content.lower():
                        relevance += 0.15
            
            # Boost for evidence-based content
            evidence_indicators = ["study", "research", "evidence", "guideline", "standard"]
            evidence_boost = sum(1 for indicator in evidence_indicators if indicator in content.lower())
            relevance += evidence_boost * 0.05
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating surgical relevance score: {e}")
            return 0.0
    
    def generate_instrument_info(self, instrument_name: str, category: str) -> Dict[str, str]:
        """Generate comprehensive instrument information using RAG"""
        try:
            instrument_info = {
                "usage_instructions": "",
                "safety_guidelines": "",
                "maintenance_info": "",
                "sterilization_requirements": "",
                "similar_instruments": []
            }
            
            # Build instrument context for RAG query
            instrument_context = {
                "name": instrument_name,
                "category": category
            }
            
            # Query for usage instructions
            usage_query = f"usage instructions for {instrument_name} {category}"
            usage_info = self.retrieve_surgical_knowledge(usage_query, instrument_context, top_k=2)
            
            for info in usage_info:
                if "usage" in info["content"].lower() or "instruction" in info["content"].lower():
                    instrument_info["usage_instructions"] += info["content"] + "\n"
            
            # Query for safety guidelines
            safety_query = f"safety guidelines for {instrument_name}"
            safety_info = self.retrieve_surgical_knowledge(safety_query, instrument_context, top_k=2)
            
            for info in safety_info:
                if "safety" in info["content"].lower() or "guideline" in info["content"].lower():
                    instrument_info["safety_guidelines"] += info["content"] + "\n"
            
            # Query for maintenance information
            maintenance_query = f"maintenance procedures for {instrument_name}"
            maintenance_info = self.retrieve_surgical_knowledge(maintenance_query, instrument_context, top_k=2)
            
            for info in maintenance_info:
                if "maintenance" in info["content"].lower() or "procedure" in info["content"].lower():
                    instrument_info["maintenance_info"] += info["content"] + "\n"
            
            # Query for sterilization requirements
            sterilization_query = f"sterilization requirements for {instrument_name}"
            sterilization_info = self.retrieve_surgical_knowledge(sterilization_query, instrument_context, top_k=2)
            
            for info in sterilization_info:
                if "sterilization" in info["content"].lower():
                    instrument_info["sterilization_requirements"] += info["content"] + "\n"
            
            # Find similar instruments
            similar_query = f"similar instruments to {instrument_name} in {category}"
            similar_info = self.retrieve_surgical_knowledge(similar_query, instrument_context, top_k=3)
            
            for info in similar_info:
                # Extract instrument names from content
                instrument_names = self.extract_instrument_names(info["content"])
                instrument_info["similar_instruments"].extend(instrument_names)
            
            # Remove duplicates and limit
            instrument_info["similar_instruments"] = list(set(instrument_info["similar_instruments"]))[:5]
            
            return instrument_info
            
        except Exception as e:
            logger.error(f"Error generating instrument info: {e}")
            return {
                "usage_instructions": "Consult manufacturer guidelines for usage instructions",
                "safety_guidelines": "Follow standard surgical safety protocols",
                "maintenance_info": "Refer to manufacturer maintenance procedures",
                "sterilization_requirements": "Follow institutional sterilization protocols",
                "similar_instruments": []
            }
    
    def extract_instrument_names(self, content: str) -> List[str]:
        """Extract instrument names from content"""
        try:
            # Simple extraction based on common instrument patterns
            instrument_names = []
            
            # Look for instrument-related patterns
            lines = content.split("\n")
            for line in lines:
                if "instrument:" in line.lower() or "tool:" in line.lower():
                    # Extract the instrument name
                    parts = line.split(":")
                    if len(parts) > 1:
                        name = parts[1].strip()
                        if name:
                            instrument_names.append(name)
            
            return instrument_names
            
        except Exception as e:
            logger.error(f"Error extracting instrument names: {e}")
            return []

class SurgicalInstrumentDataset(Dataset):
    """Custom PyTorch dataset for surgical instrument images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        return image, label

class SurgicalInstrumentCNN(nn.Module):
    """CNN model for surgical instrument recognition"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(SurgicalInstrumentCNN, self).__init__()
        
        # Use pre-trained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modify the final layer for our number of classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class YOLOModel(nn.Module):
    """YOLO-based model for object detection"""
    
    def __init__(self, num_classes: int):
        super(YOLOModel, self).__init__()
        # Simplified YOLO architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, (num_classes + 5) * 3, 1)  # 3 anchors per cell
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        features = self.features(x)
        detections = self.detection_head(features)
        return detections

class SurgicalInstrumentRecognitionSystem:
    """Main surgical instrument recognition system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = SurgicalRAGSystem(config)
        self.classification_model = None
        self.detection_model = None
        self.instrument_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all models for surgical instrument recognition"""
        try:
            # Load instrument classes
            self.load_instrument_classes()
            
            # Initialize classification model
            self.classification_model = SurgicalInstrumentCNN(
                num_classes=len(self.instrument_classes),
                pretrained=True
            ).to(self.device)
            
            # Initialize detection model
            self.detection_model = YOLOModel(
                num_classes=len(self.instrument_classes)
            ).to(self.device)
            
            # Load pre-trained weights if available
            self.load_model_weights()
            
            # Initialize image transformations
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            # Set models to evaluation mode
            self.classification_model.eval()
            self.detection_model.eval()
            
            logger.info("Surgical instrument recognition models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def load_instrument_classes(self):
        """Load surgical instrument classes"""
        try:
            # Define surgical instrument classes
            self.instrument_classes = [
                "scalpel", "forceps", "scissors", "clamp", "retractor",
                "needle_holder", "suction_tube", "electrocautery", "suture",
                "hemostat", "dilator", "probe", "cannula", "trocar",
                "laparoscope", "endoscope", "catheter", "syringe", "needle"
            ]
            
            # Create mappings
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.instrument_classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.instrument_classes)}
            
            logger.info(f"Loaded {len(self.instrument_classes)} instrument classes")
            
        except Exception as e:
            logger.error(f"Error loading instrument classes: {e}")
            raise
    
    def load_model_weights(self):
        """Load pre-trained model weights"""
        try:
            # Load classification model weights
            classification_weights_path = self.config.get("classification_weights_path")
            if classification_weights_path and os.path.exists(classification_weights_path):
                self.classification_model.load_state_dict(
                    torch.load(classification_weights_path, map_location=self.device)
                )
                logger.info("Classification model weights loaded")
            
            # Load detection model weights
            detection_weights_path = self.config.get("detection_weights_path")
            if detection_weights_path and os.path.exists(detection_weights_path):
                self.detection_model.load_state_dict(
                    torch.load(detection_weights_path, map_location=self.device)
                )
                logger.info("Detection model weights loaded")
                
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        try:
            # Apply transformations
            if self.transform:
                augmented = self.transform(image=image)
                image_tensor = augmented['image']
            else:
                # Default preprocessing
                image = cv2.resize(image, (224, 224))
                image = image / 255.0
                image = np.transpose(image, (2, 0, 1))
                image_tensor = torch.FloatTensor(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def classify_instrument(self, image: np.ndarray) -> Tuple[str, float]:
        """Classify surgical instrument in image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Perform classification
            with torch.no_grad():
                outputs = self.classification_model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get predicted class
            predicted_class = self.idx_to_class[predicted_idx.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score
            
        except Exception as e:
            logger.error(f"Error classifying instrument: {e}")
            raise
    
    def detect_instruments(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect multiple surgical instruments in image"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Perform detection
            with torch.no_grad():
                detections = self.detection_model(image_tensor)
            
            # Process detections (simplified)
            # In a real implementation, this would parse YOLO outputs
            detected_instruments = []
            
            # For now, return a simplified detection
            # This would be replaced with actual YOLO detection parsing
            height, width = image.shape[:2]
            
            # Simulate detection results
            detected_instruments.append({
                'class': 'scalpel',
                'confidence': 0.85,
                'bbox': [50, 50, 150, 200],  # [x1, y1, x2, y2]
                'center': [100, 125]
            })
            
            return detected_instruments
            
        except Exception as e:
            logger.error(f"Error detecting instruments: {e}")
            return []
    
    def recognize_instrument(self, image: np.ndarray, mode: str = 'classification') -> RecognitionResult:
        """Recognize surgical instrument in image"""
        try:
            if mode == 'classification':
                # Single instrument classification
                instrument_name, confidence_score = self.classify_instrument(image)
                
                # Get instrument information using RAG
                instrument_info = self.rag_system.generate_instrument_info(
                    instrument_name, "surgical_instrument"
                )
                
                # Generate bounding box (center of image for classification)
                height, width = image.shape[:2]
                bbox = [width//4, height//4, 3*width//4, 3*height//4]
                
            else:
                # Multi-instrument detection
                detections = self.detect_instruments(image)
                
                if not detections:
                    raise ValueError("No instruments detected")
                
                # Use the highest confidence detection
                best_detection = max(detections, key=lambda x: x['confidence'])
                instrument_name = best_detection['class']
                confidence_score = best_detection['confidence']
                bbox = best_detection['bbox']
                
                # Get instrument information using RAG
                instrument_info = self.rag_system.generate_instrument_info(
                    instrument_name, "surgical_instrument"
                )
            
            # Generate result
            result = RecognitionResult(
                instrument_id=f"inst_{instrument_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                instrument_name=instrument_name,
                category="surgical_instrument",
                confidence_score=confidence_score,
                bounding_box=bbox,
                usage_instructions=instrument_info.get('usage_instructions', ''),
                safety_guidelines=instrument_info.get('safety_guidelines', ''),
                maintenance_info=instrument_info.get('maintenance_info', ''),
                similar_instruments=instrument_info.get('similar_instruments', []),
                sterilization_status="Unknown",
                surgical_context={"instrument_name": instrument_name, "category": "surgical_instrument"},
                research_insights=self.rag_system.retrieve_surgical_knowledge(f"research evidence for {instrument_name}", {"name": instrument_name, "category": "surgical_instrument"}, top_k=3)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error recognizing instrument: {e}")
            raise
    
    def find_similar_instruments(self, instrument_name: str) -> List[str]:
        """Find similar instruments"""
        try:
            # This would typically query a database of instruments
            # For now, return related instruments based on categories
            instrument_categories = {
                "scalpel": ["surgical_knife", "blade", "lancet"],
                "forceps": ["tweezers", "clamp", "hemostat"],
                "scissors": ["surgical_scissors", "suture_scissors", "bandage_scissors"],
                "clamp": ["hemostat", "forceps", "bulldog_clamp"],
                "retractor": ["rib_spreader", "wound_retractor", "self_retaining_retractor"]
            }
            
            return instrument_categories.get(instrument_name, [])
            
        except Exception as e:
            logger.error(f"Error finding similar instruments: {e}")
            return []
    
    def analyze_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image quality for recognition"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate image quality metrics
            quality_metrics = {
                'brightness': np.mean(gray),
                'contrast': np.std(gray),
                'sharpness': self.calculate_sharpness(gray),
                'noise_level': self.estimate_noise(gray),
                'resolution': image.shape[:2],
                'aspect_ratio': image.shape[1] / image.shape[0]
            }
            
            # Determine quality score
            quality_score = self.calculate_quality_score(quality_metrics)
            quality_metrics['quality_score'] = quality_score
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            return {}
    
    def calculate_sharpness(self, gray_image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            return laplacian.var()
        except Exception as e:
            logger.error(f"Error calculating sharpness: {e}")
            return 0.0
    
    def estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in image"""
        try:
            # Apply Gaussian blur and subtract from original
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            noise = cv2.absdiff(gray_image, blurred)
            return np.mean(noise)
        except Exception as e:
            logger.error(f"Error estimating noise: {e}")
            return 0.0
    
    def calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall image quality score"""
        try:
            score = 0.0
            
            # Brightness score (optimal range: 100-200)
            brightness = metrics.get('brightness', 0)
            if 100 <= brightness <= 200:
                score += 0.3
            elif 50 <= brightness <= 250:
                score += 0.2
            else:
                score += 0.1
            
            # Contrast score (higher is better)
            contrast = metrics.get('contrast', 0)
            if contrast > 50:
                score += 0.3
            elif contrast > 30:
                score += 0.2
            else:
                score += 0.1
            
            # Sharpness score (higher is better)
            sharpness = metrics.get('sharpness', 0)
            if sharpness > 100:
                score += 0.2
            elif sharpness > 50:
                score += 0.15
            else:
                score += 0.1
            
            # Resolution score (higher is better)
            height, width = metrics.get('resolution', (0, 0))
            if height >= 512 and width >= 512:
                score += 0.2
            elif height >= 256 and width >= 256:
                score += 0.15
            else:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5

# Pydantic models for API
class RecognitionRequest(BaseModel):
    mode: str = Field(default="classification", regex="^(classification|detection)$")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class RecognitionResponse(BaseModel):
    instrument_id: str
    instrument_name: str
    category: str
    confidence_score: float
    bounding_box: List[int]
    usage_instructions: str
    safety_guidelines: str
    maintenance_info: str
    similar_instruments: List[str]
    sterilization_status: str
    image_quality: Dict[str, Any]
    timestamp: datetime

class BatchRecognitionRequest(BaseModel):
    mode: str = Field(default="detection", regex="^(classification|detection)$")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class BatchRecognitionResponse(BaseModel):
    results: List[RecognitionResponse]
    total_instruments: int
    average_confidence: float
    timestamp: datetime

# Initialize the system
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
    "chroma_db_path": "./surgical_recognition_chroma_db",
    "llm_model_name": "gpt-4",
    "classification_weights_path": "./models/surgical_classification.pth",
    "detection_weights_path": "./models/surgical_detection.pth",
    "instrument_catalogs_path": "./data/instrument_catalogs",
    "surgical_procedures_path": "./data/surgical_procedures",
    "safety_guidelines_path": "./data/safety_guidelines"
}

recognition_system = SurgicalInstrumentRecognitionSystem(config)

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_instrument(
    file: UploadFile = File(...),
    request: RecognitionRequest = RecognitionRequest()
):
    """Recognize surgical instrument in uploaded image"""
    try:
        # Read image file
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze image quality
        quality_metrics = recognition_system.analyze_image_quality(image)
        
        # Perform recognition
        result = recognition_system.recognize_instrument(image, request.mode)
        
        # Convert to response format
        response = RecognitionResponse(
            instrument_id=result.instrument_id,
            instrument_name=result.instrument_name,
            category=result.category,
            confidence_score=result.confidence_score,
            bounding_box=result.bounding_box,
            usage_instructions=result.usage_instructions,
            safety_guidelines=result.safety_guidelines,
            maintenance_info=result.maintenance_info,
            similar_instruments=result.similar_instruments,
            sterilization_status=result.sterilization_status,
            image_quality=quality_metrics,
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in recognize endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_recognize", response_model=BatchRecognitionResponse)
async def batch_recognize_instruments(
    files: List[UploadFile] = File(...),
    request: BatchRecognitionRequest = BatchRecognitionRequest()
):
    """Recognize surgical instruments in multiple images"""
    try:
        results = []
        total_confidence = 0.0
        
        for file in files:
            # Read image file
            image_data = await file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Analyze image quality
            quality_metrics = recognition_system.analyze_image_quality(image)
            
            # Perform recognition
            result = recognition_system.recognize_instrument(image, request.mode)
            
            # Convert to response format
            response = RecognitionResponse(
                instrument_id=result.instrument_id,
                instrument_name=result.instrument_name,
                category=result.category,
                confidence_score=result.confidence_score,
                bounding_box=result.bounding_box,
                usage_instructions=result.usage_instructions,
                safety_guidelines=result.safety_guidelines,
                maintenance_info=result.maintenance_info,
                similar_instruments=result.similar_instruments,
                sterilization_status=result.sterilization_status,
                image_quality=quality_metrics,
                timestamp=datetime.now()
            )
            
            results.append(response)
            total_confidence += result.confidence_score
        
        # Calculate average confidence
        average_confidence = total_confidence / len(results) if results else 0.0
        
        return BatchRecognitionResponse(
            results=results,
            total_instruments=len(results),
            average_confidence=average_confidence,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in batch recognize endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/instruments/list")
async def get_instrument_list():
    """Get list of supported surgical instruments"""
    try:
        return {
            "instruments": recognition_system.instrument_classes,
            "total_count": len(recognition_system.instrument_classes),
            "categories": {
                "cutting": ["scalpel", "scissors"],
                "grasping": ["forceps", "clamp", "hemostat"],
                "retracting": ["retractor"],
                "suturing": ["needle_holder", "suture", "needle"],
                "suction": ["suction_tube"],
                "electrosurgical": ["electrocautery"],
                "endoscopic": ["laparoscope", "endoscope"],
                "access": ["cannula", "trocar", "catheter"],
                "injection": ["syringe"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting instrument list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "Surgical Instrument Recognition API",
        "device": str(recognition_system.device),
        "models_loaded": {
            "classification": recognition_system.classification_model is not None,
            "detection": recognition_system.detection_model is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004) 