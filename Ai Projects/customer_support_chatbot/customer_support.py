"""
Generative AI Chatbot for Customer Support Automation
A comprehensive customer support system with RAG integration and multi-agent architecture
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
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
app = FastAPI(title="Customer Support Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class CustomerQuery:
    """Customer query data structure"""
    query_id: str
    customer_id: str
    query_text: str
    query_type: str
    priority: str
    category: str
    timestamp: datetime
    context: Dict[str, Any]
    support_context: Optional[str] = None
    product_info: Optional[Dict[str, Any]] = None

@dataclass
class SupportResponse:
    """Customer support response structure"""
    response_id: str
    query_id: str
    response_text: str
    confidence_score: float
    suggested_actions: List[str]
    follow_up_questions: List[str]
    escalation_needed: bool
    agent_used: str
    response_time: float
    retrieved_context: Optional[str] = None
    support_evidence: Optional[List[str]] = None

class CustomerSupportRAGSystem:
    """Enhanced RAG system for customer support knowledge retrieval"""
    
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
        self.support_databases = {}
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
            
            logger.info("Retrievers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
    
    def initialize_vector_store(self):
        """Initialize the vector store with support documents"""
        try:
            # Initialize ChromaDB with settings
            chroma_client = chromadb.PersistentClient(
                path=self.config.get("chroma_db_path", "./customer_support_chroma_db"),
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Load support documents
            self.load_support_sources()
            
            # Create vector store
            self.vector_store = ChromaDB(
                client=chroma_client,
                collection_name="support_knowledge",
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
            
            logger.info("Customer support RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def load_support_sources(self):
        """Load comprehensive customer support knowledge sources"""
        try:
            # Load FAQ documents
            self.load_faq_documents()
            
            # Load product documentation
            self.load_product_documentation()
            
            # Load troubleshooting guides
            self.load_troubleshooting_guides()
            
            # Load policy documents
            self.load_policy_documents()
            
            # Load customer feedback and reviews
            self.load_customer_feedback()
            
            # Load support ticket history
            self.load_support_ticket_history()
            
            # Load knowledge base articles
            self.load_knowledge_base_articles()
            
            # Load training materials
            self.load_training_materials()
            
            logger.info("Customer support knowledge sources loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading support sources: {e}")
    
    def load_faq_documents(self):
        """Load FAQ documents"""
        try:
            faq_path = self.config.get("faq_path", "./data/faq")
            if os.path.exists(faq_path):
                loader = DirectoryLoader(
                    faq_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                faq_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                faq_chunks = text_splitter.split_documents(faq_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(faq_chunks)
                    self.support_databases['faq'] = faq_chunks
                    
        except Exception as e:
            logger.error(f"Error loading FAQ documents: {e}")
    
    def load_product_documentation(self):
        """Load product documentation"""
        try:
            docs_path = self.config.get("product_docs_path", "./data/product_docs")
            if os.path.exists(docs_path):
                loader = DirectoryLoader(
                    docs_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                docs_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                docs_chunks = text_splitter.split_documents(docs_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(docs_chunks)
                    self.support_databases['product_docs'] = docs_chunks
                    
        except Exception as e:
            logger.error(f"Error loading product documentation: {e}")
    
    def load_troubleshooting_guides(self):
        """Load troubleshooting guides"""
        try:
            troubleshooting_path = self.config.get("troubleshooting_path", "./data/troubleshooting")
            if os.path.exists(troubleshooting_path):
                loader = DirectoryLoader(
                    troubleshooting_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                troubleshooting_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                troubleshooting_chunks = text_splitter.split_documents(troubleshooting_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(troubleshooting_chunks)
                    self.support_databases['troubleshooting'] = troubleshooting_chunks
                    
        except Exception as e:
            logger.error(f"Error loading troubleshooting guides: {e}")
    
    def load_policy_documents(self):
        """Load policy documents"""
        try:
            policies_path = self.config.get("policies_path", "./data/policies")
            if os.path.exists(policies_path):
                loader = DirectoryLoader(
                    policies_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                policies_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                policies_chunks = text_splitter.split_documents(policies_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(policies_chunks)
                    self.support_databases['policies'] = policies_chunks
                    
        except Exception as e:
            logger.error(f"Error loading policy documents: {e}")
    
    def load_customer_feedback(self):
        """Load customer feedback and reviews"""
        try:
            feedback_path = self.config.get("feedback_path", "./data/customer_feedback")
            if os.path.exists(feedback_path):
                loader = DirectoryLoader(
                    feedback_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                feedback_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                feedback_chunks = text_splitter.split_documents(feedback_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(feedback_chunks)
                    self.support_databases['customer_feedback'] = feedback_chunks
                    
        except Exception as e:
            logger.error(f"Error loading customer feedback: {e}")
    
    def load_support_ticket_history(self):
        """Load historical support ticket data"""
        try:
            tickets_path = self.config.get("tickets_path", "./data/support_tickets")
            if os.path.exists(tickets_path):
                loader = DirectoryLoader(
                    tickets_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                tickets_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                tickets_chunks = text_splitter.split_documents(tickets_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(tickets_chunks)
                    self.support_databases['support_tickets'] = tickets_chunks
                    
        except Exception as e:
            logger.error(f"Error loading support ticket history: {e}")
    
    def load_knowledge_base_articles(self):
        """Load knowledge base articles"""
        try:
            kb_path = self.config.get("knowledge_base_path", "./data/knowledge_base")
            if os.path.exists(kb_path):
                loader = DirectoryLoader(
                    kb_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                kb_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                kb_chunks = text_splitter.split_documents(kb_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(kb_chunks)
                    self.support_databases['knowledge_base'] = kb_chunks
                    
        except Exception as e:
            logger.error(f"Error loading knowledge base articles: {e}")
    
    def load_training_materials(self):
        """Load support agent training materials"""
        try:
            training_path = self.config.get("training_path", "./data/training_materials")
            if os.path.exists(training_path):
                loader = DirectoryLoader(
                    training_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                training_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                training_chunks = text_splitter.split_documents(training_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(training_chunks)
                    self.support_databases['training_materials'] = training_chunks
                    
        except Exception as e:
            logger.error(f"Error loading training materials: {e}")
    
    def retrieve_support_knowledge(self, query: str, customer_context: Optional[Dict[str, Any]] = None, use_ensemble: bool = True) -> List[str]:
        """Retrieve relevant support knowledge with enhanced context awareness"""
        try:
            # Check cache first
            cache_key = f"{query}_{hash(str(customer_context))}"
            if cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            # Enhance query with customer context
            enhanced_query = self.enhance_query_with_context(query, customer_context)
            
            if use_ensemble and hasattr(self, 'ensemble_retriever'):
                docs = self.ensemble_retriever.get_relevant_documents(enhanced_query)
            else:
                docs = self.retriever.get_relevant_documents(enhanced_query)
            
            # Filter and rank results
            filtered_docs = self.filter_relevant_documents(docs, query, customer_context)
            
            # Cache results
            self.retrieval_cache[cache_key] = [doc.page_content for doc in filtered_docs]
            
            return [doc.page_content for doc in filtered_docs]
            
        except Exception as e:
            logger.error(f"Error retrieving support knowledge: {e}")
            return []
    
    def enhance_query_with_context(self, query: str, customer_context: Optional[Dict[str, Any]] = None) -> str:
        """Enhance query with customer context"""
        try:
            if not customer_context:
                return query
            
            enhanced_parts = [query]
            
            # Add product information
            if 'product' in customer_context:
                enhanced_parts.append(f"Product: {customer_context['product']}")
            
            # Add customer tier
            if 'customer_tier' in customer_context:
                enhanced_parts.append(f"Customer Tier: {customer_context['customer_tier']}")
            
            # Add previous issues
            if 'previous_issues' in customer_context:
                enhanced_parts.append(f"Previous Issues: {', '.join(customer_context['previous_issues'])}")
            
            # Add support history
            if 'support_history' in customer_context:
                enhanced_parts.append(f"Support History: {customer_context['support_history']}")
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query with context: {e}")
            return query
    
    def filter_relevant_documents(self, docs: List, query: str, customer_context: Optional[Dict[str, Any]] = None) -> List:
        """Filter and rank documents based on relevance"""
        try:
            if not docs:
                return []
            
            # Calculate relevance scores
            scored_docs = []
            for doc in docs:
                relevance_score = self.calculate_support_relevance_score(doc.page_content, query, customer_context)
                scored_docs.append((doc, relevance_score))
            
            # Sort by relevance score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top documents
            return [doc for doc, score in scored_docs[:5]]
            
        except Exception as e:
            logger.error(f"Error filtering documents: {e}")
            return docs[:5]
    
    def calculate_support_relevance_score(self, content: str, query: str, customer_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate relevance score for support content"""
        try:
            score = 0.0
            
            # Basic text similarity
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if query_words:
                word_overlap = len(query_words.intersection(content_words)) / len(query_words)
                score += word_overlap * 0.4
            
            # Context relevance
            if customer_context:
                if 'product' in customer_context and customer_context['product'].lower() in content.lower():
                    score += 0.3
                
                if 'customer_tier' in customer_context and customer_context['customer_tier'].lower() in content.lower():
                    score += 0.2
                
                if 'previous_issues' in customer_context:
                    for issue in customer_context['previous_issues']:
                        if issue.lower() in content.lower():
                            score += 0.1
            
            # Content type preference
            if any(keyword in content.lower() for keyword in ['solution', 'fix', 'resolve', 'answer']):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5
    
    def get_support_context(self, query: str, customer_context: Optional[Dict[str, Any]] = None) -> str:
        """Get comprehensive support context"""
        try:
            knowledge = self.retrieve_support_knowledge(query, customer_context)
            
            if not knowledge:
                return "No specific support information found."
            
            # Combine knowledge into context
            context_parts = []
            for i, info in enumerate(knowledge[:3], 1):
                context_parts.append(f"Support Information {i}: {info}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting support context: {e}")
            return "Unable to retrieve support context."
    
    def extract_support_entities(self, query: str) -> Dict[str, Any]:
        """Extract support-related entities from query"""
        try:
            entities = {
                'product_mentions': [],
                'issue_types': [],
                'urgency_indicators': [],
                'action_requests': []
            }
            
            query_lower = query.lower()
            
            # Extract product mentions
            product_keywords = ['app', 'software', 'platform', 'service', 'tool', 'system']
            for keyword in product_keywords:
                if keyword in query_lower:
                    entities['product_mentions'].append(keyword)
            
            # Extract issue types
            issue_keywords = ['error', 'bug', 'problem', 'issue', 'broken', 'not working']
            for keyword in issue_keywords:
                if keyword in query_lower:
                    entities['issue_types'].append(keyword)
            
            # Extract urgency indicators
            urgency_keywords = ['urgent', 'emergency', 'asap', 'critical', 'important']
            for keyword in urgency_keywords:
                if keyword in query_lower:
                    entities['urgency_indicators'].append(keyword)
            
            # Extract action requests
            action_keywords = ['help', 'support', 'assist', 'guide', 'explain', 'show']
            for keyword in action_keywords:
                if keyword in query_lower:
                    entities['action_requests'].append(keyword)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting support entities: {e}")
            return {}

class SupportAgent:
    """Base class for support agents"""
    
    def __init__(self, name: str, rag_system: CustomerSupportRAGSystem, config: Dict[str, Any]):
        self.name = name
        self.rag_system = rag_system
        self.config = config
        self.llm = ChatOpenAI(
            model_name=config.get("llm_model_name", "gpt-4"),
            temperature=0.1,
            openai_api_key=config.get("openai_api_key")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def process_query(self, query: CustomerQuery) -> SupportResponse:
        """Process customer query and generate response"""
        raise NotImplementedError("Subclasses must implement process_query")

class GeneralSupportAgent(SupportAgent):
    """General support agent for common queries"""
    
    def process_query(self, query: CustomerQuery) -> SupportResponse:
        """Process general support queries"""
        try:
            start_time = datetime.now()
            
            # Retrieve relevant knowledge
            knowledge = self.rag_system.retrieve_support_knowledge(query.query_text)
            
            # Generate response using LLM
            prompt_template = PromptTemplate(
                input_variables=["query", "knowledge", "customer_context"],
                template="""
                You are a helpful customer support agent. Based on the customer query and available knowledge, 
                provide a helpful and accurate response.
                
                Customer Query: {query}
                Available Knowledge: {knowledge}
                Customer Context: {customer_context}
                
                Provide a clear, helpful response that addresses the customer's question or concern.
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response_text = chain.run({
                "query": query.query_text,
                "knowledge": "\n".join(knowledge),
                "customer_context": str(query.context)
            })
            
            # Calculate confidence score
            confidence_score = self.calculate_confidence(query.query_text, knowledge)
            
            # Generate suggested actions
            suggested_actions = self.generate_suggested_actions(query, response_text)
            
            # Generate follow-up questions
            follow_up_questions = self.generate_follow_up_questions(query, response_text)
            
            # Determine if escalation is needed
            escalation_needed = self.determine_escalation(query, confidence_score)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            return SupportResponse(
                response_id=f"resp_{query.query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                query_id=query.query_id,
                response_text=response_text,
                confidence_score=confidence_score,
                suggested_actions=suggested_actions,
                follow_up_questions=follow_up_questions,
                escalation_needed=escalation_needed,
                agent_used=self.name,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Error in general support agent: {e}")
            raise
    
    def calculate_confidence(self, query: str, knowledge: List[str]) -> float:
        """Calculate confidence score for response"""
        try:
            if not knowledge:
                return 0.3
            
            # Simple confidence calculation based on knowledge relevance
            # In a real system, this would be more sophisticated
            relevance_score = len(knowledge) / 5.0  # Normalize to 0-1
            return min(relevance_score, 0.9)  # Cap at 0.9
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def generate_suggested_actions(self, query: CustomerQuery, response: str) -> List[str]:
        """Generate suggested actions for customer"""
        try:
            # This would typically analyze the query and response to suggest actions
            # For now, return generic suggestions
            return [
                "Review the provided information",
                "Contact support if you need further assistance",
                "Check our FAQ section for similar questions"
            ]
            
        except Exception as e:
            logger.error(f"Error generating suggested actions: {e}")
            return []
    
    def generate_follow_up_questions(self, query: CustomerQuery, response: str) -> List[str]:
        """Generate follow-up questions"""
        try:
            # This would typically analyze the conversation to generate relevant follow-ups
            # For now, return generic questions
            return [
                "Is there anything else I can help you with?",
                "Did this answer your question completely?",
                "Would you like me to provide more details on any specific aspect?"
            ]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []
    
    def determine_escalation(self, query: CustomerQuery, confidence: float) -> bool:
        """Determine if escalation to human agent is needed"""
        try:
            # Escalate if confidence is low or query is complex
            if confidence < 0.5:
                return True
            
            # Check for escalation keywords
            escalation_keywords = [
                "complaint", "refund", "cancel", "urgent", "emergency",
                "manager", "supervisor", "escalate", "unhappy", "dissatisfied"
            ]
            
            query_lower = query.query_text.lower()
            if any(keyword in query_lower for keyword in escalation_keywords):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining escalation: {e}")
            return False

class TechnicalSupportAgent(SupportAgent):
    """Technical support agent for technical issues"""
    
    def process_query(self, query: CustomerQuery) -> SupportResponse:
        """Process technical support queries"""
        try:
            start_time = datetime.now()
            
            # Retrieve technical knowledge
            knowledge = self.rag_system.retrieve_support_knowledge(query.query_text)
            
            # Generate technical response
            prompt_template = PromptTemplate(
                input_variables=["query", "knowledge", "customer_context"],
                template="""
                You are a technical support specialist. Provide detailed technical assistance based on the customer's issue.
                
                Customer Query: {query}
                Technical Knowledge: {knowledge}
                Customer Context: {customer_context}
                
                Provide step-by-step technical guidance and troubleshooting steps.
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response_text = chain.run({
                "query": query.query_text,
                "knowledge": "\n".join(knowledge),
                "customer_context": str(query.context)
            })
            
            # Calculate confidence score
            confidence_score = self.calculate_technical_confidence(query.query_text, knowledge)
            
            # Generate technical actions
            suggested_actions = self.generate_technical_actions(query, response_text)
            
            # Generate technical follow-ups
            follow_up_questions = self.generate_technical_follow_ups(query, response_text)
            
            # Determine escalation
            escalation_needed = self.determine_technical_escalation(query, confidence_score)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            return SupportResponse(
                response_id=f"resp_{query.query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                query_id=query.query_id,
                response_text=response_text,
                confidence_score=confidence_score,
                suggested_actions=suggested_actions,
                follow_up_questions=follow_up_questions,
                escalation_needed=escalation_needed,
                agent_used=self.name,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Error in technical support agent: {e}")
            raise
    
    def calculate_technical_confidence(self, query: str, knowledge: List[str]) -> float:
        """Calculate confidence for technical responses"""
        try:
            if not knowledge:
                return 0.2
            
            # Technical queries often require more specific knowledge
            relevance_score = len(knowledge) / 5.0
            return min(relevance_score * 0.8, 0.85)  # Slightly lower confidence for technical issues
            
        except Exception as e:
            logger.error(f"Error calculating technical confidence: {e}")
            return 0.4
    
    def generate_technical_actions(self, query: CustomerQuery, response: str) -> List[str]:
        """Generate technical actions"""
        try:
            return [
                "Follow the troubleshooting steps provided",
                "Check system requirements and compatibility",
                "Try the suggested solutions in order",
                "Contact technical support if issues persist"
            ]
            
        except Exception as e:
            logger.error(f"Error generating technical actions: {e}")
            return []
    
    def generate_technical_follow_ups(self, query: CustomerQuery, response: str) -> List[str]:
        """Generate technical follow-up questions"""
        try:
            return [
                "Did the troubleshooting steps resolve your issue?",
                "What error messages are you seeing?",
                "What operating system and version are you using?",
                "Have you tried restarting your system?"
            ]
            
        except Exception as e:
            logger.error(f"Error generating technical follow-ups: {e}")
            return []
    
    def determine_technical_escalation(self, query: CustomerQuery, confidence: float) -> bool:
        """Determine if technical escalation is needed"""
        try:
            # Escalate for complex technical issues
            if confidence < 0.6:
                return True
            
            # Check for complex technical keywords
            complex_keywords = [
                "crash", "error", "bug", "broken", "not working",
                "advanced", "complex", "custom", "integration", "api"
            ]
            
            query_lower = query.query_text.lower()
            if any(keyword in query_lower for keyword in complex_keywords):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining technical escalation: {e}")
            return True

class BillingSupportAgent(SupportAgent):
    """Billing support agent for billing and payment issues"""
    
    def process_query(self, query: CustomerQuery) -> SupportResponse:
        """Process billing support queries"""
        try:
            start_time = datetime.now()
            
            # Retrieve billing knowledge
            knowledge = self.rag_system.retrieve_support_knowledge(query.query_text)
            
            # Generate billing response
            prompt_template = PromptTemplate(
                input_variables=["query", "knowledge", "customer_context"],
                template="""
                You are a billing support specialist. Help customers with billing, payment, and account-related questions.
                
                Customer Query: {query}
                Billing Knowledge: {knowledge}
                Customer Context: {customer_context}
                
                Provide clear information about billing policies, payment options, and account management.
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response_text = chain.run({
                "query": query.query_text,
                "knowledge": "\n".join(knowledge),
                "customer_context": str(query.context)
            })
            
            # Calculate confidence score
            confidence_score = self.calculate_billing_confidence(query.query_text, knowledge)
            
            # Generate billing actions
            suggested_actions = self.generate_billing_actions(query, response_text)
            
            # Generate billing follow-ups
            follow_up_questions = self.generate_billing_follow_ups(query, response_text)
            
            # Determine escalation
            escalation_needed = self.determine_billing_escalation(query, confidence_score)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            return SupportResponse(
                response_id=f"resp_{query.query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                query_id=query.query_id,
                response_text=response_text,
                confidence_score=confidence_score,
                suggested_actions=suggested_actions,
                follow_up_questions=follow_up_questions,
                escalation_needed=escalation_needed,
                agent_used=self.name,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Error in billing support agent: {e}")
            raise
    
    def calculate_billing_confidence(self, query: str, knowledge: List[str]) -> float:
        """Calculate confidence for billing responses"""
        try:
            if not knowledge:
                return 0.2
            
            # Billing queries often require policy knowledge
            relevance_score = len(knowledge) / 5.0
            return min(relevance_score * 0.9, 0.9)
            
        except Exception as e:
            logger.error(f"Error calculating billing confidence: {e}")
            return 0.4
    
    def generate_billing_actions(self, query: CustomerQuery, response: str) -> List[str]:
        """Generate billing actions"""
        try:
            return [
                "Review your billing statement",
                "Update your payment method if needed",
                "Check your account settings",
                "Contact billing support for account-specific issues"
            ]
            
        except Exception as e:
            logger.error(f"Error generating billing actions: {e}")
            return []
    
    def generate_billing_follow_ups(self, query: CustomerQuery, response: str) -> List[str]:
        """Generate billing follow-up questions"""
        try:
            return [
                "Do you need help updating your payment information?",
                "Would you like to review your billing history?",
                "Do you have questions about our refund policy?",
                "Would you like to set up automatic payments?"
            ]
            
        except Exception as e:
            logger.error(f"Error generating billing follow-ups: {e}")
            return []
    
    def determine_billing_escalation(self, query: CustomerQuery, confidence: float) -> bool:
        """Determine if billing escalation is needed"""
        try:
            # Escalate for sensitive billing issues
            if confidence < 0.7:
                return True
            
            # Check for sensitive billing keywords
            sensitive_keywords = [
                "refund", "dispute", "chargeback", "fraud", "unauthorized",
                "cancel", "terminate", "legal", "complaint", "escalate"
            ]
            
            query_lower = query.query_text.lower()
            if any(keyword in query_lower for keyword in sensitive_keywords):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining billing escalation: {e}")
            return True

class CustomerSupportOrchestrator:
    """Orchestrator for managing multiple support agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = CustomerSupportRAGSystem(config)
        self.agents = {}
        self.query_classifier = None
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize all support agents"""
        try:
            # Initialize different types of agents
            self.agents['general'] = GeneralSupportAgent(
                "General Support Agent",
                self.rag_system,
                self.config
            )
            
            self.agents['technical'] = TechnicalSupportAgent(
                "Technical Support Agent",
                self.rag_system,
                self.config
            )
            
            self.agents['billing'] = BillingSupportAgent(
                "Billing Support Agent",
                self.rag_system,
                self.config
            )
            
            # Initialize query classifier
            self.initialize_query_classifier()
            
            logger.info("Customer support agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    def initialize_query_classifier(self):
        """Initialize query classification system"""
        try:
            # Simple keyword-based classifier
            # In a real system, this would be a trained ML model
            self.query_classifier = {
                'technical': [
                    'error', 'bug', 'crash', 'not working', 'broken',
                    'install', 'update', 'download', 'compatibility',
                    'performance', 'slow', 'freeze', 'technical'
                ],
                'billing': [
                    'payment', 'billing', 'charge', 'invoice', 'refund',
                    'subscription', 'cancel', 'upgrade', 'downgrade',
                    'credit card', 'paypal', 'billing', 'account'
                ],
                'general': [
                    'how to', 'what is', 'where is', 'when', 'why',
                    'help', 'support', 'question', 'information'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error initializing query classifier: {e}")
    
    def classify_query(self, query_text: str) -> str:
        """Classify query type"""
        try:
            query_lower = query_text.lower()
            
            # Count keyword matches for each category
            scores = {}
            for category, keywords in self.query_classifier.items():
                score = sum(1 for keyword in keywords if keyword in query_lower)
                scores[category] = score
            
            # Return category with highest score
            if scores:
                return max(scores, key=scores.get)
            else:
                return 'general'
                
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return 'general'
    
    def select_agent(self, query: CustomerQuery) -> SupportAgent:
        """Select appropriate agent for query"""
        try:
            # Classify query type
            query_type = self.classify_query(query.query_text)
            
            # Select agent based on classification
            if query_type in self.agents:
                return self.agents[query_type]
            else:
                return self.agents['general']
                
        except Exception as e:
            logger.error(f"Error selecting agent: {e}")
            return self.agents['general']
    
    def process_customer_query(self, query: CustomerQuery) -> SupportResponse:
        """Process customer query using appropriate agent"""
        try:
            # Select appropriate agent
            agent = self.select_agent(query)
            
            # Process query with selected agent
            response = agent.process_query(query)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing customer query: {e}")
            raise
    
    def update_conversation_history(self, query: CustomerQuery, response: SupportResponse):
        """Update conversation history"""
        try:
            # This would typically store conversation history in a database
            # For now, just log the interaction
            logger.info(f"Conversation updated: Query {query.query_id} -> Response {response.response_id}")
            
        except Exception as e:
            logger.error(f"Error updating conversation history: {e}")

# Pydantic models for API
class CustomerQueryRequest(BaseModel):
    customer_id: str
    query_text: str
    query_type: Optional[str] = None
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    category: Optional[str] = None
    context: Dict[str, Any] = {}

class SupportResponseResponse(BaseModel):
    response_id: str
    query_id: str
    response_text: str
    confidence_score: float
    suggested_actions: List[str]
    follow_up_questions: List[str]
    escalation_needed: bool
    agent_used: str
    response_time: float
    timestamp: datetime

class ConversationRequest(BaseModel):
    customer_id: str
    messages: List[Dict[str, str]]
    context: Dict[str, Any] = {}

class ConversationResponse(BaseModel):
    conversation_id: str
    responses: List[SupportResponseResponse]
    total_responses: int
    average_confidence: float
    escalation_recommended: bool
    timestamp: datetime

# Initialize the system
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
    "chroma_db_path": "./customer_support_chroma_db",
    "llm_model_name": "gpt-4",
    "faq_path": "./data/faq",
    "product_docs_path": "./data/product_docs",
    "troubleshooting_path": "./data/troubleshooting",
    "policies_path": "./data/policies"
}

support_orchestrator = CustomerSupportOrchestrator(config)

@app.post("/chat", response_model=SupportResponseResponse)
async def chat_with_support(request: CustomerQueryRequest):
    """Chat with customer support"""
    try:
        # Create customer query
        query = CustomerQuery(
            query_id=f"query_{request.customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            customer_id=request.customer_id,
            query_text=request.query_text,
            query_type=request.query_type or "general",
            priority=request.priority,
            category=request.category or "general",
            timestamp=datetime.now(),
            context=request.context
        )
        
        # Process query
        response = support_orchestrator.process_customer_query(query)
        
        # Update conversation history
        support_orchestrator.update_conversation_history(query, response)
        
        # Convert to response format
        response_response = SupportResponseResponse(
            response_id=response.response_id,
            query_id=response.query_id,
            response_text=response.response_text,
            confidence_score=response.confidence_score,
            suggested_actions=response.suggested_actions,
            follow_up_questions=response.follow_up_questions,
            escalation_needed=response.escalation_needed,
            agent_used=response.agent_used,
            response_time=response.response_time,
            timestamp=datetime.now()
        )
        
        return response_response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation", response_model=ConversationResponse)
async def handle_conversation(request: ConversationRequest):
    """Handle multi-turn conversation"""
    try:
        responses = []
        total_confidence = 0.0
        escalation_recommended = False
        
        # Process each message in the conversation
        for message in request.messages:
            # Create query for each message
            query = CustomerQuery(
                query_id=f"query_{request.customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                customer_id=request.customer_id,
                query_text=message.get('text', ''),
                query_type=message.get('type', 'general'),
                priority=message.get('priority', 'normal'),
                category=message.get('category', 'general'),
                timestamp=datetime.now(),
                context=request.context
            )
            
            # Process query
            response = support_orchestrator.process_customer_query(query)
            
            # Update conversation history
            support_orchestrator.update_conversation_history(query, response)
            
            # Convert to response format
            response_response = SupportResponseResponse(
                response_id=response.response_id,
                query_id=response.query_id,
                response_text=response.response_text,
                confidence_score=response.confidence_score,
                suggested_actions=response.suggested_actions,
                follow_up_questions=response.follow_up_questions,
                escalation_needed=response.escalation_needed,
                agent_used=response.agent_used,
                response_time=response.response_time,
                timestamp=datetime.now()
            )
            
            responses.append(response_response)
            total_confidence += response.confidence_score
            
            # Check for escalation
            if response.escalation_needed:
                escalation_recommended = True
        
        # Calculate average confidence
        average_confidence = total_confidence / len(responses) if responses else 0.0
        
        return ConversationResponse(
            conversation_id=f"conv_{request.customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            responses=responses,
            total_responses=len(responses),
            average_confidence=average_confidence,
            escalation_recommended=escalation_recommended,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in conversation endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/list")
async def get_agents_list():
    """Get list of available support agents"""
    try:
        return {
            "agents": list(support_orchestrator.agents.keys()),
            "agent_descriptions": {
                "general": "Handles general customer inquiries and questions",
                "technical": "Specializes in technical issues and troubleshooting",
                "billing": "Handles billing, payment, and account-related issues"
            },
            "total_agents": len(support_orchestrator.agents)
        }
    except Exception as e:
        logger.error(f"Error getting agents list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "Customer Support Chatbot API",
        "agents_loaded": len(support_orchestrator.agents),
        "rag_system_ready": support_orchestrator.rag_system.retriever is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 