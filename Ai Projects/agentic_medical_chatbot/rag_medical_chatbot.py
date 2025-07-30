"""
Agentic RAG-Based Medical Chatbot Implementation
A comprehensive medical chatbot system with RAG integration and multi-agent architecture
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Core AI/ML Libraries
import openai
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import BaseOutputParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ChromaDB, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.merger_retriever import MergerRetriever

# Medical and Healthcare Libraries
import spacy
from transformers import pipeline
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Web Framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
from config import (
    OPENAI_API_KEY, 
    PINECONE_API_KEY, 
    CHROMA_DB_PATH,
    MEDICAL_KNOWLEDGE_BASE_PATH,
    RAG_PARAMS,
    KNOWLEDGE_SOURCES
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Agentic Medical Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class MedicalQuery:
    """Data class for medical queries"""
    user_id: str
    query_text: str
    query_type: str  # symptom, medication, treatment, emergency
    urgency_level: int  # 1-5 scale
    context: Dict[str, Any]
    timestamp: datetime

@dataclass
class MedicalResponse:
    """Data class for medical responses"""
    response_text: str
    confidence_score: float
    sources: List[str]
    recommendations: List[str]
    escalation_needed: bool
    follow_up_questions: List[str]
    retrieved_context: Dict[str, Any]

class MedicalRAGSystem:
    """Enhanced RAG system for medical knowledge retrieval with multiple retrieval strategies"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.bm25_retriever = None
        self.knowledge_base = {}
        self.retrieval_cache = {}
        self.initialize_knowledge_base()
        self.setup_retrievers()
    
    def initialize_knowledge_base(self):
        """Initialize medical knowledge base with RAG capabilities"""
        try:
            # Initialize vector store
            if PINECONE_API_KEY:
                import pinecone
                pinecone.init(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
                self.vector_store = Pinecone.from_existing_index(
                    index_name="medical-knowledge",
                    embedding=self.embeddings
                )
            else:
                self.vector_store = ChromaDB(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=self.embeddings
                )
            
            # Load medical knowledge sources
            self.load_medical_sources()
            logger.info("Medical knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            raise
    
    def setup_retrievers(self):
        """Setup multiple retrieval strategies for enhanced information retrieval"""
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
            
            logger.info("Retrievers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
    
    def load_medical_sources(self):
        """Load medical knowledge sources from various repositories"""
        try:
            # Load from local medical knowledge base
            if os.path.exists(MEDICAL_KNOWLEDGE_BASE_PATH):
                self.load_medical_documents("local")
            
            # Load from PubMed abstracts (if available)
            self.load_pubmed_articles()
            
            # Load from clinical guidelines
            self.load_clinical_guidelines()
            
            # Load from drug databases
            self.load_drug_databases()
            
            # Load from medical textbooks
            self.load_medical_textbooks()
            
            logger.info(f"Loaded {len(self.knowledge_base)} medical knowledge sources")
            
        except Exception as e:
            logger.error(f"Error loading medical sources: {e}")
    
    def load_medical_documents(self, source_type: str) -> List[str]:
        """Load medical documents from specified source"""
        documents = []
        source_path = KNOWLEDGE_SOURCES.get(source_type, "")
        
        if os.path.exists(source_path):
            try:
                if source_type == "local":
                    loader = DirectoryLoader(
                        source_path,
                        glob="**/*.pdf",
                        loader_cls=PyPDFLoader
                    )
                    documents = loader.load()
                    
                    # Add to knowledge base
                    for doc in documents:
                        self.knowledge_base[doc.metadata.get("source", "unknown")] = {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "type": source_type
                        }
                
                logger.info(f"Loaded {len(documents)} documents from {source_type}")
                
            except Exception as e:
                logger.error(f"Error loading documents from {source_type}: {e}")
        
        return documents
    
    def load_pubmed_articles(self):
        """Load medical articles from PubMed API"""
        try:
            # This would integrate with PubMed API
            # For now, we'll simulate with sample data
            sample_articles = [
                {
                    "title": "Recent advances in cardiovascular disease treatment",
                    "abstract": "This review covers the latest developments...",
                    "authors": ["Dr. Smith", "Dr. Johnson"],
                    "journal": "Journal of Cardiology",
                    "year": 2024
                }
            ]
            
            for article in sample_articles:
                self.knowledge_base[f"pubmed_{article['title']}"] = {
                    "content": f"{article['title']}\n{article['abstract']}",
                    "metadata": article,
                    "type": "pubmed"
                }
                
        except Exception as e:
            logger.error(f"Error loading PubMed articles: {e}")
    
    def load_clinical_guidelines(self):
        """Load clinical practice guidelines"""
        try:
            # Load from clinical guidelines databases
            guidelines = [
                {
                    "title": "Hypertension Management Guidelines",
                    "content": "Blood pressure should be monitored regularly...",
                    "source": "American Heart Association",
                    "year": 2024
                }
            ]
            
            for guideline in guidelines:
                self.knowledge_base[f"guideline_{guideline['title']}"] = {
                    "content": guideline['content'],
                    "metadata": guideline,
                    "type": "guideline"
                }
                
        except Exception as e:
            logger.error(f"Error loading clinical guidelines: {e}")
    
    def load_drug_databases(self):
        """Load drug information databases"""
        try:
            # Load drug interaction and safety data
            drug_data = [
                {
                    "name": "Aspirin",
                    "interactions": ["Warfarin", "Ibuprofen"],
                    "side_effects": ["Stomach upset", "Bleeding risk"],
                    "dosage": "81-325mg daily"
                }
            ]
            
            for drug in drug_data:
                self.knowledge_base[f"drug_{drug['name']}"] = {
                    "content": f"Drug: {drug['name']}\nInteractions: {', '.join(drug['interactions'])}\nSide effects: {', '.join(drug['side_effects'])}",
                    "metadata": drug,
                    "type": "drug"
                }
                
        except Exception as e:
            logger.error(f"Error loading drug databases: {e}")
    
    def load_medical_textbooks(self):
        """Load medical textbook content"""
        try:
            # Load from medical textbook repositories
            textbooks = [
                {
                    "title": "Harrison's Principles of Internal Medicine",
                    "chapter": "Cardiovascular Diseases",
                    "content": "Cardiovascular diseases are the leading cause...",
                    "edition": "21st Edition"
                }
            ]
            
            for textbook in textbooks:
                self.knowledge_base[f"textbook_{textbook['title']}_{textbook['chapter']}"] = {
                    "content": textbook['content'],
                    "metadata": textbook,
                    "type": "textbook"
                }
                
        except Exception as e:
            logger.error(f"Error loading medical textbooks: {e}")
    
    def retrieve_relevant_info(self, query: str, top_k: int = 5, 
                             use_ensemble: bool = True) -> List[Dict]:
        """Enhanced retrieval with multiple strategies and caching"""
        try:
            # Check cache first
            cache_key = f"{query}_{top_k}_{use_ensemble}"
            if cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            retrieved_info = []
            
            if use_ensemble and hasattr(self, 'ensemble_retriever'):
                # Use ensemble retriever
                docs = self.ensemble_retriever.get_relevant_documents(query)
            else:
                # Use vector store directly
                docs = self.vector_store.similarity_search(query, k=top_k)
            
            # Process retrieved documents
            for doc in docs:
                info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": self.calculate_relevance_score(query, doc.page_content)
                }
                retrieved_info.append(info)
            
            # Sort by relevance score
            retrieved_info.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Cache results
            self.retrieval_cache[cache_key] = retrieved_info[:top_k]
            
            logger.info(f"Retrieved {len(retrieved_info)} relevant documents for query: {query[:50]}...")
            return retrieved_info[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving information: {e}")
            return []
    
    def calculate_relevance_score(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        try:
            # Simple TF-IDF based relevance scoring
            query_words = set(query.lower().split())
            content_words = content.lower().split()
            
            # Calculate word overlap
            overlap = len(query_words.intersection(set(content_words)))
            total_query_words = len(query_words)
            
            if total_query_words == 0:
                return 0.0
            
            relevance = overlap / total_query_words
            
            # Boost score for medical terms
            medical_terms = ["symptom", "treatment", "medication", "diagnosis", "disease"]
            medical_boost = sum(1 for term in medical_terms if term in content.lower())
            relevance += medical_boost * 0.1
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0
    
    def get_medical_context(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant medical context for the query"""
        try:
            context = {
                "user_medical_history": user_context.get("medical_history", []),
                "current_medications": user_context.get("medications", []),
                "allergies": user_context.get("allergies", []),
                "age": user_context.get("age"),
                "gender": user_context.get("gender"),
                "relevant_conditions": []
            }
            
            # Extract medical conditions from query
            medical_conditions = self.extract_medical_entities(query)
            context["relevant_conditions"] = medical_conditions
            
            # Get relevant medical guidelines
            if medical_conditions:
                guidelines = self.retrieve_relevant_info(
                    f"guidelines for {' '.join(medical_conditions)}", 
                    top_k=3
                )
                context["relevant_guidelines"] = guidelines
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting medical context: {e}")
            return {}
    
    def extract_medical_entities(self, text: str) -> List[str]:
        """Extract medical entities from text using NLP"""
        try:
            # Load spaCy model for medical entity extraction
            if not hasattr(self, 'nlp'):
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except:
                    # Fallback to basic entity extraction
                    medical_terms = [
                        "diabetes", "hypertension", "asthma", "arthritis",
                        "cancer", "heart disease", "stroke", "depression"
                    ]
                    return [term for term in medical_terms if term in text.lower()]
            
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in ["CONDITION", "DISEASE", "SYMPTOM"]]
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting medical entities: {e}")
            return []

class MedicalAgent:
    """Base class for medical agents"""
    
    def __init__(self, agent_type: str, rag_system: MedicalRAGSystem):
        self.agent_type = agent_type
        self.rag_system = rag_system
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        self.context = {}
    
    async def process_query(self, query: MedicalQuery) -> MedicalResponse:
        """Process medical query and generate response"""
        raise NotImplementedError
    
    def update_context(self, new_context: Dict[str, Any]):
        """Update agent context"""
        self.context.update(new_context)

class SymptomAnalysisAgent(MedicalAgent):
    """Agent for analyzing symptoms and providing guidance"""
    
    def __init__(self, rag_system: MedicalRAGSystem):
        super().__init__("symptom_analysis", rag_system)
        self.symptom_classifier = pipeline(
            "text-classification",
            model="medical-bert-symptom-classifier"
        )
    
    async def process_query(self, query: MedicalQuery) -> MedicalResponse:
        """Analyze symptoms and provide guidance"""
        
        # Retrieve relevant symptom information
        symptom_info = self.rag_system.retrieve_relevant_info(
            f"symptoms {query.query_text}", top_k=3
        )
        
        # Classify symptom severity
        severity = self.classify_symptom_severity(query.query_text)
        
        # Generate response
        response_text = self.generate_symptom_response(
            query.query_text, symptom_info, severity
        )
        
        # Determine if escalation is needed
        escalation_needed = severity >= 4 or query.urgency_level >= 4
        
        return MedicalResponse(
            response_text=response_text,
            confidence_score=0.85,
            sources=[info["source"] for info in symptom_info],
            recommendations=self.generate_recommendations(symptom_info),
            escalation_needed=escalation_needed,
            follow_up_questions=self.generate_follow_up_questions(query.query_text)
        )
    
    def classify_symptom_severity(self, symptom_text: str) -> int:
        """Classify symptom severity on a 1-5 scale"""
        # Implement symptom severity classification
        severe_keywords = ["severe", "intense", "unbearable", "emergency"]
        moderate_keywords = ["moderate", "noticeable", "concerning"]
        
        text_lower = symptom_text.lower()
        
        if any(keyword in text_lower for keyword in severe_keywords):
            return 5
        elif any(keyword in text_lower for keyword in moderate_keywords):
            return 3
        else:
            return 2
    
    def generate_symptom_response(self, symptom: str, info: List[Dict], severity: int) -> str:
        """Generate symptom analysis response"""
        response_template = """
        Based on your symptoms: {symptom}
        
        Severity Level: {severity}/5
        
        Analysis: {analysis}
        
        Recommendations: {recommendations}
        
        {escalation_note}
        """
        
        analysis = " ".join([item["content"][:200] for item in info])
        recommendations = self.generate_recommendations(info)
        escalation_note = "Please seek immediate medical attention." if severity >= 4 else ""
        
        return response_template.format(
            symptom=symptom,
            severity=severity,
            analysis=analysis,
            recommendations=", ".join(recommendations),
            escalation_note=escalation_note
        )
    
    def generate_recommendations(self, info: List[Dict]) -> List[str]:
        """Generate recommendations based on symptom information"""
        recommendations = [
            "Monitor your symptoms closely",
            "Keep a symptom diary",
            "Stay hydrated and rest"
        ]
        
        # Add specific recommendations based on retrieved information
        for item in info:
            if "rest" in item["content"].lower():
                recommendations.append("Get adequate rest")
            if "hydration" in item["content"].lower():
                recommendations.append("Maintain proper hydration")
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_follow_up_questions(self, symptom: str) -> List[str]:
        """Generate follow-up questions for better symptom understanding"""
        questions = [
            "How long have you been experiencing these symptoms?",
            "Are the symptoms constant or intermittent?",
            "Have you noticed any triggers that make symptoms worse?",
            "Are you currently taking any medications?"
        ]
        
        # Add specific questions based on symptom type
        if "pain" in symptom.lower():
            questions.append("On a scale of 1-10, how would you rate the pain?")
        if "fever" in symptom.lower():
            questions.append("What is your current body temperature?")
        
        return questions

class MedicationInformationAgent(MedicalAgent):
    """Agent for providing medication information"""
    
    def __init__(self, rag_system: MedicalRAGSystem):
        super().__init__("medication_info", rag_system)
        self.drug_interaction_checker = self.load_drug_interaction_model()
    
    async def process_query(self, query: MedicalQuery) -> MedicalResponse:
        """Provide medication information and check interactions"""
        
        # Extract medication names from query
        medications = self.extract_medications(query.query_text)
        
        # Retrieve medication information
        med_info = []
        for med in medications:
            info = self.rag_system.retrieve_relevant_info(
                f"medication {med} dosage side effects", top_k=2
            )
            med_info.extend(info)
        
        # Check drug interactions
        interactions = self.check_drug_interactions(medications)
        
        # Generate response
        response_text = self.generate_medication_response(
            medications, med_info, interactions
        )
        
        return MedicalResponse(
            response_text=response_text,
            confidence_score=0.90,
            sources=[info["source"] for info in med_info],
            recommendations=self.generate_medication_recommendations(med_info, interactions),
            escalation_needed=len(interactions) > 0,
            follow_up_questions=self.generate_medication_questions(medications)
        )
    
    def extract_medications(self, text: str) -> List[str]:
        """Extract medication names from text using NLP"""
        # Use spaCy for named entity recognition
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        medications = []
        for ent in doc.ents:
            if ent.label_ in ["DRUG", "CHEMICAL"]:
                medications.append(ent.text)
        
        return medications
    
    def load_drug_interaction_model(self):
        """Load drug interaction checking model"""
        # This would be a pre-trained model for drug interactions
        # For now, return a simple function
        return lambda meds: self.simple_interaction_check(meds)
    
    def simple_interaction_check(self, medications: List[str]) -> List[str]:
        """Simple drug interaction checker"""
        # Known interactions (in a real system, this would be a comprehensive database)
        known_interactions = {
            ("warfarin", "aspirin"): "Increased bleeding risk",
            ("simvastatin", "grapefruit"): "Increased statin levels",
            ("digoxin", "diuretics"): "Electrolyte imbalance risk"
        }
        
        interactions = []
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                pair = tuple(sorted([med1.lower(), med2.lower()]))
                if pair in known_interactions:
                    interactions.append(known_interactions[pair])
        
        return interactions
    
    def check_drug_interactions(self, medications: List[str]) -> List[str]:
        """Check for drug interactions"""
        return self.drug_interaction_checker(medications)
    
    def generate_medication_response(self, medications: List[str], info: List[Dict], interactions: List[str]) -> str:
        """Generate medication information response"""
        response_template = """
        Medication Information for: {medications}
        
        {medication_details}
        
        {interaction_warnings}
        
        Important Notes: {notes}
        """
        
        medication_details = "\n".join([
            f"- {item['content'][:300]}..." for item in info[:3]
        ])
        
        interaction_warnings = ""
        if interactions:
            interaction_warnings = "⚠️ DRUG INTERACTIONS DETECTED:\n" + "\n".join([
                f"- {interaction}" for interaction in interactions
            ])
        
        return response_template.format(
            medications=", ".join(medications),
            medication_details=medication_details,
            interaction_warnings=interaction_warnings,
            notes="Always consult with a healthcare provider before starting or stopping medications."
        )
    
    def generate_medication_recommendations(self, info: List[Dict], interactions: List[str]) -> List[str]:
        """Generate medication-related recommendations"""
        recommendations = [
            "Take medications as prescribed",
            "Keep a medication list updated",
            "Store medications properly"
        ]
        
        if interactions:
            recommendations.append("Consult healthcare provider about drug interactions")
            recommendations.append("Monitor for side effects closely")
        
        return recommendations
    
    def generate_medication_questions(self, medications: List[str]) -> List[str]:
        """Generate follow-up questions about medications"""
        questions = [
            "Are you experiencing any side effects?",
            "Are you taking the medication as prescribed?",
            "Have you missed any doses recently?"
        ]
        
        return questions

class MedicalChatbotOrchestrator:
    """Main orchestrator for the medical chatbot system"""
    
    def __init__(self):
        self.rag_system = MedicalRAGSystem()
        self.agents = {
            "symptom": SymptomAnalysisAgent(self.rag_system),
            "medication": MedicationInformationAgent(self.rag_system),
            "treatment": TreatmentRecommendationAgent(self.rag_system),
            "emergency": EmergencyAssessmentAgent(self.rag_system)
        }
        self.conversation_history = {}
    
    async def process_medical_query(self, query: MedicalQuery) -> MedicalResponse:
        """Process medical query using appropriate agent"""
        
        # Determine which agent to use
        agent = self.select_agent(query)
        
        # Update agent context with conversation history
        if query.user_id in self.conversation_history:
            agent.update_context({
                "conversation_history": self.conversation_history[query.user_id]
            })
        
        # Process query
        response = await agent.process_query(query)
        
        # Update conversation history
        self.update_conversation_history(query, response)
        
        return response
    
    def select_agent(self, query: MedicalQuery) -> MedicalAgent:
        """Select appropriate agent based on query type"""
        if query.query_type in self.agents:
            return self.agents[query.query_type]
        else:
            # Default to symptom analysis
            return self.agents["symptom"]
    
    def update_conversation_history(self, query: MedicalQuery, response: MedicalResponse):
        """Update conversation history for the user"""
        if query.user_id not in self.conversation_history:
            self.conversation_history[query.user_id] = []
        
        self.conversation_history[query.user_id].append({
            "query": query.query_text,
            "response": response.response_text,
            "timestamp": query.timestamp,
            "escalation_needed": response.escalation_needed
        })
        
        # Keep only last 10 interactions
        if len(self.conversation_history[query.user_id]) > 10:
            self.conversation_history[query.user_id] = self.conversation_history[query.user_id][-10:]

# Pydantic models for API
class MedicalQueryRequest(BaseModel):
    user_id: str
    query_text: str
    query_type: str = "symptom"
    urgency_level: int = 1
    context: Dict[str, Any] = {}

class MedicalQueryResponse(BaseModel):
    response_text: str
    confidence_score: float
    sources: List[str]
    recommendations: List[str]
    escalation_needed: bool
    follow_up_questions: List[str]

# Initialize orchestrator
chatbot_orchestrator = MedicalChatbotOrchestrator()

@app.post("/chat", response_model=MedicalQueryResponse)
async def chat_endpoint(request: MedicalQueryRequest):
    """Main chat endpoint for medical queries"""
    try:
        # Create medical query object
        query = MedicalQuery(
            user_id=request.user_id,
            query_text=request.query_text,
            query_type=request.query_type,
            urgency_level=request.urgency_level,
            context=request.context,
            timestamp=datetime.now()
        )
        
        # Process query
        response = await chatbot_orchestrator.process_medical_query(query)
        
        return MedicalQueryResponse(
            response_text=response.response_text,
            confidence_score=response.confidence_score,
            sources=response.sources,
            recommendations=response.recommendations,
            escalation_needed=response.escalation_needed,
            follow_up_questions=response.follow_up_questions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 