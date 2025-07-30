"""
Retail Customer Segmentation and Purchase Prediction System
A comprehensive system for customer segmentation and purchase prediction using ML and RAG
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
app = FastAPI(title="Retail Customer Segmentation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class CustomerData:
    """Customer data structure for segmentation"""
    customer_id: str
    age: int
    gender: str
    income: float
    location: str
    purchase_history: List[Dict[str, Any]]
    browsing_history: List[Dict[str, Any]]
    preferences: Dict[str, Any]
    loyalty_score: float
    total_spent: float
    avg_order_value: float
    frequency_score: float
    recency_score: float
    category_preferences: Dict[str, float]
    seasonal_patterns: Dict[str, float]
    market_context: Optional[Dict[str, Any]] = None
    industry_trends: Optional[List[str]] = None

@dataclass
class SegmentationResult:
    """Result of customer segmentation"""
    customer_id: str
    segment_id: int
    segment_name: str
    segment_description: str
    segment_characteristics: Dict[str, Any]
    purchase_prediction: float
    next_purchase_date: datetime
    recommended_products: List[str]
    marketing_strategy: str
    confidence_score: float
    market_insights: Optional[List[str]] = None
    trend_analysis: Optional[Dict[str, str]] = None

class RetailRAGSystem:
    """Enhanced RAG system for retail knowledge retrieval"""
    
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
        self.retail_databases = {}
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
            
            logger.info("Retail analytics retrievers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
    
    def initialize_vector_store(self):
        """Initialize the vector store with retail documents"""
        try:
            # Initialize ChromaDB with settings
            chroma_client = chromadb.PersistentClient(
                path=self.config.get("chroma_db_path", "./retail_analytics_chroma_db"),
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Load retail documents
            self.load_retail_sources()
            
            # Create vector store
            self.vector_store = ChromaDB(
                client=chroma_client,
                collection_name="retail_knowledge",
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
            
            logger.info("Retail RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def load_retail_sources(self):
        """Load comprehensive retail knowledge sources"""
        try:
            # Load market research
            self.load_market_research()
            
            # Load customer behavior studies
            self.load_customer_behavior_studies()
            
            # Load marketing strategies
            self.load_marketing_strategies()
            
            # Load industry trends
            self.load_industry_trends()
            
            # Load competitive analysis
            self.load_competitive_analysis()
            
            # Load product catalogs
            self.load_product_catalogs()
            
            # Load seasonal patterns
            self.load_seasonal_patterns()
            
            # Load pricing strategies
            self.load_pricing_strategies()
            
            logger.info("Retail knowledge sources loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading retail sources: {e}")
    
    def load_market_research(self):
        """Load market research documents"""
        try:
            research_path = self.config.get("market_research_path", "./data/market_research")
            if os.path.exists(research_path):
                loader = DirectoryLoader(
                    research_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                research_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                research_chunks = text_splitter.split_documents(research_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(research_chunks)
                    self.retail_databases['market_research'] = research_chunks
                    
        except Exception as e:
            logger.error(f"Error loading market research: {e}")
    
    def load_customer_behavior_studies(self):
        """Load customer behavior studies"""
        try:
            behavior_path = self.config.get("behavior_studies_path", "./data/behavior_studies")
            if os.path.exists(behavior_path):
                loader = DirectoryLoader(
                    behavior_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                behavior_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                behavior_chunks = text_splitter.split_documents(behavior_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(behavior_chunks)
                    self.retail_databases['behavior_studies'] = behavior_chunks
                    
        except Exception as e:
            logger.error(f"Error loading customer behavior studies: {e}")
    
    def load_marketing_strategies(self):
        """Load marketing strategies"""
        try:
            marketing_path = self.config.get("marketing_strategies_path", "./data/marketing_strategies")
            if os.path.exists(marketing_path):
                loader = DirectoryLoader(
                    marketing_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                marketing_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                marketing_chunks = text_splitter.split_documents(marketing_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(marketing_chunks)
                    self.retail_databases['marketing_strategies'] = marketing_chunks
                    
        except Exception as e:
            logger.error(f"Error loading marketing strategies: {e}")
    
    def load_industry_trends(self):
        """Load industry trends and analysis"""
        try:
            trends_path = self.config.get("industry_trends_path", "./data/industry_trends")
            if os.path.exists(trends_path):
                loader = DirectoryLoader(
                    trends_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                trends_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                trends_chunks = text_splitter.split_documents(trends_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(trends_chunks)
                    self.retail_databases['industry_trends'] = trends_chunks
                    
        except Exception as e:
            logger.error(f"Error loading industry trends: {e}")
    
    def load_competitive_analysis(self):
        """Load competitive analysis documents"""
        try:
            competitive_path = self.config.get("competitive_analysis_path", "./data/competitive_analysis")
            if os.path.exists(competitive_path):
                loader = DirectoryLoader(
                    competitive_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                competitive_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                competitive_chunks = text_splitter.split_documents(competitive_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(competitive_chunks)
                    self.retail_databases['competitive_analysis'] = competitive_chunks
                    
        except Exception as e:
            logger.error(f"Error loading competitive analysis: {e}")
    
    def load_product_catalogs(self):
        """Load product catalogs and descriptions"""
        try:
            catalog_path = self.config.get("product_catalog_path", "./data/product_catalog")
            if os.path.exists(catalog_path):
                loader = DirectoryLoader(
                    catalog_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                catalog_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                catalog_chunks = text_splitter.split_documents(catalog_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(catalog_chunks)
                    self.retail_databases['product_catalog'] = catalog_chunks
                    
        except Exception as e:
            logger.error(f"Error loading product catalogs: {e}")
    
    def load_seasonal_patterns(self):
        """Load seasonal patterns and analysis"""
        try:
            seasonal_path = self.config.get("seasonal_patterns_path", "./data/seasonal_patterns")
            if os.path.exists(seasonal_path):
                loader = DirectoryLoader(
                    seasonal_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                seasonal_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                seasonal_chunks = text_splitter.split_documents(seasonal_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(seasonal_chunks)
                    self.retail_databases['seasonal_patterns'] = seasonal_chunks
                    
        except Exception as e:
            logger.error(f"Error loading seasonal patterns: {e}")
    
    def load_pricing_strategies(self):
        """Load pricing strategies and analysis"""
        try:
            pricing_path = self.config.get("pricing_strategies_path", "./data/pricing_strategies")
            if os.path.exists(pricing_path):
                loader = DirectoryLoader(
                    pricing_path,
                    glob="**/*.txt",
                    loader_cls=TextLoader
                )
                pricing_docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                pricing_chunks = text_splitter.split_documents(pricing_docs)
                
                if self.vector_store:
                    self.vector_store.add_documents(pricing_chunks)
                    self.retail_databases['pricing_strategies'] = pricing_chunks
                    
        except Exception as e:
            logger.error(f"Error loading pricing strategies: {e}")
    
    def retrieve_retail_knowledge(self, query: str, customer_context: Optional[Dict[str, Any]] = None, use_ensemble: bool = True) -> List[str]:
        """Retrieve relevant retail knowledge with enhanced context awareness"""
        try:
            # Check cache first
            cache_key = f"{query}_{hash(str(customer_context))}"
            if cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            # Enhance query with customer context
            enhanced_query = self.enhance_query_with_customer_context(query, customer_context)
            
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
            logger.error(f"Error retrieving retail knowledge: {e}")
            return []
    
    def enhance_query_with_customer_context(self, query: str, customer_context: Optional[Dict[str, Any]] = None) -> str:
        """Enhance query with customer context"""
        try:
            if not customer_context:
                return query
            
            enhanced_parts = [query]
            
            # Add demographic information
            if 'age' in customer_context:
                enhanced_parts.append(f"Age: {customer_context['age']}")
            
            if 'income' in customer_context:
                enhanced_parts.append(f"Income: ${customer_context['income']}")
            
            if 'location' in customer_context:
                enhanced_parts.append(f"Location: {customer_context['location']}")
            
            # Add behavioral information
            if 'total_spent' in customer_context:
                enhanced_parts.append(f"Total Spent: ${customer_context['total_spent']}")
            
            if 'loyalty_score' in customer_context:
                enhanced_parts.append(f"Loyalty Score: {customer_context['loyalty_score']}")
            
            # Add preference information
            if 'category_preferences' in customer_context:
                top_categories = sorted(
                    customer_context['category_preferences'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                enhanced_parts.append(f"Top Categories: {', '.join([cat for cat, _ in top_categories])}")
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query with customer context: {e}")
            return query
    
    def filter_relevant_documents(self, docs: List, query: str, customer_context: Optional[Dict[str, Any]] = None) -> List:
        """Filter and rank documents based on relevance"""
        try:
            if not docs:
                return []
            
            # Calculate relevance scores
            scored_docs = []
            for doc in docs:
                relevance_score = self.calculate_retail_relevance_score(doc.page_content, query, customer_context)
                scored_docs.append((doc, relevance_score))
            
            # Sort by relevance score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top documents
            return [doc for doc, score in scored_docs[:5]]
            
        except Exception as e:
            logger.error(f"Error filtering documents: {e}")
            return docs[:5]
    
    def calculate_retail_relevance_score(self, content: str, query: str, customer_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate relevance score for retail content"""
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
                if 'income' in customer_context:
                    income = customer_context['income']
                    if income > 100000 and 'premium' in content.lower():
                        score += 0.2
                    elif income < 50000 and 'budget' in content.lower():
                        score += 0.2
                
                if 'age' in customer_context:
                    age = customer_context['age']
                    if age < 30 and 'young' in content.lower():
                        score += 0.1
                    elif age > 50 and 'mature' in content.lower():
                        score += 0.1
                
                if 'category_preferences' in customer_context:
                    for category in customer_context['category_preferences']:
                        if category.lower() in content.lower():
                            score += 0.1
            
            # Content type preference
            if any(keyword in content.lower() for keyword in ['strategy', 'trend', 'analysis', 'recommendation']):
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating retail relevance score: {e}")
            return 0.5
    
    def get_retail_context(self, query: str, customer_context: Optional[Dict[str, Any]] = None) -> str:
        """Get comprehensive retail context"""
        try:
            knowledge = self.retrieve_retail_knowledge(query, customer_context)
            
            if not knowledge:
                return "No specific retail information found."
            
            # Combine knowledge into context
            context_parts = []
            for i, info in enumerate(knowledge[:3], 1):
                context_parts.append(f"Retail Information {i}: {info}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting retail context: {e}")
            return "Unable to retrieve retail context."
    
    def extract_retail_entities(self, query: str) -> Dict[str, Any]:
        """Extract retail-related entities from query"""
        try:
            entities = {
                'customer_segments': [],
                'marketing_channels': [],
                'product_categories': [],
                'pricing_strategies': []
            }
            
            query_lower = query.lower()
            
            # Extract customer segments
            segment_keywords = ['premium', 'budget', 'loyal', 'new', 'occasional', 'high-value']
            for keyword in segment_keywords:
                if keyword in query_lower:
                    entities['customer_segments'].append(keyword)
            
            # Extract marketing channels
            channel_keywords = ['email', 'social', 'digital', 'traditional', 'mobile', 'web']
            for keyword in channel_keywords:
                if keyword in query_lower:
                    entities['marketing_channels'].append(keyword)
            
            # Extract product categories
            category_keywords = ['electronics', 'clothing', 'home', 'beauty', 'sports', 'books']
            for keyword in category_keywords:
                if keyword in query_lower:
                    entities['product_categories'].append(keyword)
            
            # Extract pricing strategies
            pricing_keywords = ['discount', 'premium', 'competitive', 'dynamic', 'bundling']
            for keyword in pricing_keywords:
                if keyword in query_lower:
                    entities['pricing_strategies'].append(keyword)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting retail entities: {e}")
            return {}
    
    def generate_marketing_strategy(self, customer_data: CustomerData, segment_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate enhanced marketing strategy using RAG"""
        try:
            # Create customer context
            customer_context = {
                'age': customer_data.age,
                'income': customer_data.income,
                'location': customer_data.location,
                'total_spent': customer_data.total_spent,
                'loyalty_score': customer_data.loyalty_score,
                'category_preferences': customer_data.category_preferences
            }
            
            # Create query for marketing strategy
            query = f"""
            Customer segmentation and marketing strategy:
            Customer ID: {customer_data.customer_id}
            Age: {customer_data.age}
            Income: ${customer_data.income}
            Total Spent: ${customer_data.total_spent}
            Segment: {segment_info.get('name', '')}
            Segment Characteristics: {segment_info.get('characteristics', '')}
            
            Generate personalized marketing strategy and product recommendations.
            """
            
            # Retrieve relevant knowledge
            knowledge = self.retrieve_retail_knowledge(query, customer_context)
            
            # Generate strategy using LLM
            prompt_template = PromptTemplate(
                input_variables=["customer_info", "segment_info", "knowledge", "customer_context"],
                template="""
                Based on the following customer information, retail knowledge, and customer context, 
                generate a comprehensive personalized marketing strategy:
                
                Customer Information: {customer_info}
                Customer Context: {customer_context}
                Segment Information: {segment_info}
                Retail Knowledge: {knowledge}
                
                Provide:
                1. Marketing Strategy
                2. Product Recommendations
                3. Communication Approach
                4. Pricing Strategy
                5. Channel Strategy
                6. Timing and Frequency
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            response = chain.run({
                "customer_info": str(customer_data.__dict__),
                "customer_context": str(customer_context),
                "segment_info": str(segment_info),
                "knowledge": "\n".join(knowledge)
            })
            
            # Parse response into sections
            sections = response.split('\n\n')
            strategy = {}
            
            for section in sections:
                if 'marketing' in section.lower():
                    strategy['marketing_strategy'] = section.strip()
                elif 'product' in section.lower():
                    strategy['product_recommendations'] = section.strip()
                elif 'communication' in section.lower():
                    strategy['communication_approach'] = section.strip()
                elif 'pricing' in section.lower():
                    strategy['pricing_strategy'] = section.strip()
                elif 'channel' in section.lower():
                    strategy['channel_strategy'] = section.strip()
                elif 'timing' in section.lower():
                    strategy['timing_frequency'] = section.strip()
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating marketing strategy: {e}")
            return {
                'marketing_strategy': 'Standard marketing approach',
                'product_recommendations': 'General product recommendations',
                'communication_approach': 'Standard communication channels',
                'pricing_strategy': 'Competitive pricing strategy',
                'channel_strategy': 'Multi-channel approach',
                'timing_frequency': 'Regular engagement schedule'
            }

class CustomerSegmentationDataset(Dataset):
    """Custom PyTorch dataset for customer segmentation"""
    
    def __init__(self, customers: List[CustomerData], labels: List[int], transform=None):
        self.customers = customers
        self.labels = labels
        self.transform = transform
        self.feature_extractor = CustomerFeatureExtractor()
        self.processed_data = self.preprocess_data()
    
    def preprocess_data(self) -> np.ndarray:
        """Preprocess customer data for segmentation"""
        processed_data = []
        
        for customer in self.customers:
            features = self.feature_extractor.extract_features(customer)
            processed_data.append(features)
        
        return np.array(processed_data)
    
    def __len__(self):
        return len(self.customers)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.processed_data[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label

class CustomerFeatureExtractor:
    """Extract features from customer data for segmentation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def extract_features(self, customer: CustomerData) -> List[float]:
        """Extract numerical features from customer"""
        features = []
        
        # Basic demographic features
        features.extend([
            customer.age,
            customer.income,
            customer.loyalty_score,
            customer.total_spent,
            customer.avg_order_value,
            customer.frequency_score,
            customer.recency_score
        ])
        
        # Gender encoding
        gender_encoded = self.encode_categorical(customer.gender, 'gender')
        features.append(gender_encoded)
        
        # Location encoding
        location_encoded = self.encode_categorical(customer.location, 'location')
        features.append(location_encoded)
        
        # Purchase history features
        purchase_features = self.extract_purchase_features(customer.purchase_history)
        features.extend(purchase_features)
        
        # Category preference features
        category_features = self.extract_category_features(customer.category_preferences)
        features.extend(category_features)
        
        # Seasonal pattern features
        seasonal_features = self.extract_seasonal_features(customer.seasonal_patterns)
        features.extend(seasonal_features)
        
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
    
    def extract_purchase_features(self, purchase_history: List[Dict[str, Any]]) -> List[float]:
        """Extract features from purchase history"""
        try:
            if not purchase_history:
                return [0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Calculate purchase statistics
            total_purchases = len(purchase_history)
            total_amount = sum(p.get('amount', 0) for p in purchase_history)
            avg_amount = total_amount / total_purchases if total_purchases > 0 else 0
            
            # Calculate purchase frequency
            if len(purchase_history) > 1:
                dates = [p.get('date', datetime.now()) for p in purchase_history]
                dates.sort()
                time_span = (dates[-1] - dates[0]).days
                frequency = total_purchases / (time_span + 1)  # Add 1 to avoid division by zero
            else:
                frequency = 0.0
            
            # Calculate variety (number of unique categories)
            categories = set(p.get('category', '') for p in purchase_history)
            variety = len(categories)
            
            # Calculate recency (days since last purchase)
            if purchase_history:
                last_purchase = max(p.get('date', datetime.now()) for p in purchase_history)
                recency = (datetime.now() - last_purchase).days
            else:
                recency = 365  # Default to 1 year if no purchases
            
            return [total_purchases, total_amount, avg_amount, frequency, variety, recency]
            
        except Exception as e:
            logger.error(f"Error extracting purchase features: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def extract_category_features(self, category_preferences: Dict[str, float]) -> List[float]:
        """Extract features from category preferences"""
        try:
            if not category_preferences:
                return [0.0] * 5  # Default features
            
            # Get top categories by preference
            sorted_categories = sorted(category_preferences.items(), key=lambda x: x[1], reverse=True)
            
            # Extract top 5 category preferences
            features = []
            for i in range(5):
                if i < len(sorted_categories):
                    features.append(sorted_categories[i][1])
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting category features: {e}")
            return [0.0] * 5
    
    def extract_seasonal_features(self, seasonal_patterns: Dict[str, float]) -> List[float]:
        """Extract features from seasonal patterns"""
        try:
            if not seasonal_patterns:
                return [0.0] * 4  # Default features for 4 seasons
            
            # Extract seasonal preferences
            features = []
            seasons = ['spring', 'summer', 'fall', 'winter']
            
            for season in seasons:
                features.append(seasonal_patterns.get(season, 0.0))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting seasonal features: {e}")
            return [0.0] * 4

class PurchasePredictionModel(nn.Module):
    """Neural network for purchase prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        super(PurchasePredictionModel, self).__init__()
        
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
        
        # Output layer for purchase amount prediction
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class RetailAnalyticsSystem:
    """Main retail analytics system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = RetailRAGSystem(config)
        self.segmentation_models = {}
        self.prediction_models = {}
        self.feature_extractor = CustomerFeatureExtractor()
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segment_descriptions = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all retail analytics models"""
        try:
            # Initialize segmentation models
            self.segmentation_models = {
                'kmeans': KMeans(
                    n_clusters=self.config.get('num_segments', 5),
                    random_state=42,
                    n_init=10
                ),
                'dbscan': DBSCAN(
                    eps=self.config.get('dbscan_eps', 0.5),
                    min_samples=self.config.get('dbscan_min_samples', 5)
                ),
                'hierarchical': AgglomerativeClustering(
                    n_clusters=self.config.get('num_segments', 5),
                    linkage='ward'
                )
            }
            
            # Initialize prediction models
            self.prediction_models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42
                ),
                'linear_regression': LinearRegression()
            }
            
            # Initialize neural network for purchase prediction
            input_dim = self.config.get('input_dim', 25)  # Number of features
            hidden_dims = self.config.get('hidden_dims', [64, 32, 16])
            
            self.prediction_models['neural_network'] = PurchasePredictionModel(
                input_dim=input_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
            
            # Load pre-trained models if available
            self.load_model_weights()
            
            # Initialize segment descriptions
            self.initialize_segment_descriptions()
            
            logger.info("Retail analytics models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def load_model_weights(self):
        """Load pre-trained model weights"""
        try:
            # Load segmentation models
            for model_name in ['kmeans', 'dbscan', 'hierarchical']:
                model_path = self.config.get(f"{model_name}_model_path")
                if model_path and os.path.exists(model_path):
                    self.segmentation_models[model_name] = joblib.load(model_path)
                    logger.info(f"{model_name} segmentation model loaded")
            
            # Load prediction models
            for model_name in ['random_forest', 'gradient_boosting', 'linear_regression']:
                model_path = self.config.get(f"{model_name}_model_path")
                if model_path and os.path.exists(model_path):
                    self.prediction_models[model_name] = joblib.load(model_path)
                    logger.info(f"{model_name} prediction model loaded")
            
            # Load neural network weights
            nn_path = self.config.get("neural_network_weights_path")
            if nn_path and os.path.exists(nn_path):
                self.prediction_models['neural_network'].load_state_dict(
                    torch.load(nn_path, map_location=self.device)
                )
                logger.info("Neural network weights loaded")
                
        except Exception as e:
            logger.error(f"Error loading model weights: {e}")
    
    def initialize_segment_descriptions(self):
        """Initialize segment descriptions"""
        try:
            self.segment_descriptions = {
                0: {
                    'name': 'High-Value Loyalists',
                    'description': 'High-income customers with strong loyalty and frequent purchases',
                    'characteristics': {
                        'income_level': 'High',
                        'purchase_frequency': 'High',
                        'loyalty': 'Very High',
                        'avg_order_value': 'High'
                    }
                },
                1: {
                    'name': 'Budget Conscious',
                    'description': 'Price-sensitive customers who look for deals and discounts',
                    'characteristics': {
                        'income_level': 'Medium-Low',
                        'purchase_frequency': 'Medium',
                        'loyalty': 'Medium',
                        'avg_order_value': 'Low'
                    }
                },
                2: {
                    'name': 'Occasional Shoppers',
                    'description': 'Infrequent customers with moderate spending patterns',
                    'characteristics': {
                        'income_level': 'Medium',
                        'purchase_frequency': 'Low',
                        'loyalty': 'Low',
                        'avg_order_value': 'Medium'
                    }
                },
                3: {
                    'name': 'Premium Customers',
                    'description': 'High-income customers who prefer premium products',
                    'characteristics': {
                        'income_level': 'Very High',
                        'purchase_frequency': 'Medium',
                        'loyalty': 'High',
                        'avg_order_value': 'Very High'
                    }
                },
                4: {
                    'name': 'New Customers',
                    'description': 'Recently acquired customers with potential for growth',
                    'characteristics': {
                        'income_level': 'Variable',
                        'purchase_frequency': 'Low',
                        'loyalty': 'Unknown',
                        'avg_order_value': 'Variable'
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error initializing segment descriptions: {e}")
    
    def extract_customer_features(self, customer: CustomerData) -> np.ndarray:
        """Extract features from customer data"""
        try:
            features = self.feature_extractor.extract_features(customer)
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features = self.scaler.fit_transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting customer features: {e}")
            raise
    
    def segment_customer(self, customer: CustomerData, method: str = 'kmeans') -> Tuple[int, Dict[str, Any]]:
        """Segment customer using specified method"""
        try:
            # Extract features
            features = self.extract_customer_features(customer)
            
            # Perform segmentation
            if method not in self.segmentation_models:
                raise ValueError(f"Unknown segmentation method: {method}")
            
            model = self.segmentation_models[method]
            segment_id = model.fit_predict(features)[0]
            
            # Get segment information
            segment_info = self.segment_descriptions.get(segment_id, {
                'name': f'Segment {segment_id}',
                'description': 'Customer segment',
                'characteristics': {}
            })
            
            return int(segment_id), segment_info
            
        except Exception as e:
            logger.error(f"Error segmenting customer: {e}")
            raise
    
    def predict_purchase(self, customer: CustomerData, model_name: str = 'ensemble') -> Tuple[float, float]:
        """Predict next purchase amount using specified model"""
        try:
            # Extract features
            features = self.extract_customer_features(customer)
            
            if model_name == 'ensemble':
                # Use ensemble of models
                predictions = []
                weights = [0.4, 0.3, 0.2, 0.1]  # Weights for different models
                
                for i, (model_key, weight) in enumerate(zip(['random_forest', 'gradient_boosting', 'linear_regression', 'neural_network'], weights)):
                    if model_key in self.prediction_models:
                        model = self.prediction_models[model_key]
                        
                        if model_key == 'neural_network':
                            # Use neural network
                            model.eval()
                            with torch.no_grad():
                                features_tensor = torch.FloatTensor(features).to(self.device)
                                prediction = model(features_tensor).item()
                        else:
                            # Use traditional ML model
                            prediction = model.predict(features)[0]
                        
                        predictions.append(prediction * weight)
                
                predicted_amount = sum(predictions)
                confidence_score = 0.8  # Ensemble confidence
                
            else:
                # Use specific model
                if model_name in self.prediction_models:
                    model = self.prediction_models[model_name]
                    
                    if model_name == 'neural_network':
                        # Use neural network
                        model.eval()
                        with torch.no_grad():
                            features_tensor = torch.FloatTensor(features).to(self.device)
                            predicted_amount = model(features_tensor).item()
                    else:
                        # Use traditional ML model
                        predicted_amount = model.predict(features)[0]
                    
                    confidence_score = 0.75
                else:
                    raise ValueError(f"Unknown model: {model_name}")
            
            return predicted_amount, confidence_score
            
        except Exception as e:
            logger.error(f"Error predicting purchase: {e}")
            raise
    
    def predict_next_purchase_date(self, customer: CustomerData) -> datetime:
        """Predict next purchase date"""
        try:
            # Simple prediction based on purchase frequency
            if customer.purchase_history:
                # Calculate average days between purchases
                dates = [p.get('date', datetime.now()) for p in customer.purchase_history]
                dates.sort()
                
                if len(dates) > 1:
                    intervals = []
                    for i in range(1, len(dates)):
                        interval = (dates[i] - dates[i-1]).days
                        intervals.append(interval)
                    
                    avg_interval = sum(intervals) / len(intervals)
                    next_date = datetime.now() + timedelta(days=avg_interval)
                else:
                    # If only one purchase, predict 30 days from now
                    next_date = datetime.now() + timedelta(days=30)
            else:
                # If no purchase history, predict 60 days from now
                next_date = datetime.now() + timedelta(days=60)
            
            return next_date
            
        except Exception as e:
            logger.error(f"Error predicting next purchase date: {e}")
            return datetime.now() + timedelta(days=30)
    
    def generate_product_recommendations(self, customer: CustomerData, segment_info: Dict[str, Any]) -> List[str]:
        """Generate product recommendations"""
        try:
            # This would typically use a recommendation system
            # For now, return recommendations based on segment and preferences
            
            recommendations = []
            
            # Get top category preferences
            top_categories = sorted(
                customer.category_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for category, score in top_categories:
                if score > 0.5:  # Only recommend if preference is high enough
                    recommendations.append(f"Premium {category} products")
                    recommendations.append(f"New {category} arrivals")
            
            # Add segment-specific recommendations
            segment_name = segment_info.get('name', '').lower()
            if 'high-value' in segment_name or 'premium' in segment_name:
                recommendations.extend([
                    "Luxury items",
                    "Exclusive collections",
                    "VIP services"
                ])
            elif 'budget' in segment_name:
                recommendations.extend([
                    "Sale items",
                    "Discount offers",
                    "Value packs"
                ])
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating product recommendations: {e}")
            return ["Popular products", "New arrivals", "Best sellers"]
    
    def analyze_customer(self, customer: CustomerData) -> SegmentationResult:
        """Analyze customer for segmentation and prediction"""
        try:
            # Perform customer segmentation
            segment_id, segment_info = self.segment_customer(customer)
            
            # Predict next purchase
            predicted_amount, confidence_score = self.predict_purchase(customer)
            
            # Predict next purchase date
            next_purchase_date = self.predict_next_purchase_date(customer)
            
            # Generate product recommendations
            recommended_products = self.generate_product_recommendations(customer, segment_info)
            
            # Generate marketing strategy using RAG
            marketing_strategy = self.rag_system.generate_marketing_strategy(customer, segment_info)
            
            return SegmentationResult(
                customer_id=customer.customer_id,
                segment_id=segment_id,
                segment_name=segment_info.get('name', f'Segment {segment_id}'),
                segment_description=segment_info.get('description', ''),
                segment_characteristics=segment_info.get('characteristics', {}),
                purchase_prediction=predicted_amount,
                next_purchase_date=next_purchase_date,
                recommended_products=recommended_products,
                marketing_strategy=marketing_strategy.get('marketing_strategy', 'Standard approach'),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing customer: {e}")
            raise

# Pydantic models for API
class CustomerDataRequest(BaseModel):
    customer_id: str
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., regex="^(M|F|Other)$")
    income: float = Field(..., ge=0)
    location: str
    purchase_history: List[Dict[str, Any]] = []
    browsing_history: List[Dict[str, Any]] = []
    preferences: Dict[str, Any] = {}
    loyalty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_spent: float = Field(default=0.0, ge=0.0)
    avg_order_value: float = Field(default=0.0, ge=0.0)
    frequency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    recency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    category_preferences: Dict[str, float] = {}
    seasonal_patterns: Dict[str, float] = {}

class SegmentationResponse(BaseModel):
    customer_id: str
    segment_id: int
    segment_name: str
    segment_description: str
    segment_characteristics: Dict[str, Any]
    purchase_prediction: float
    next_purchase_date: datetime
    recommended_products: List[str]
    marketing_strategy: str
    confidence_score: float
    timestamp: datetime

class BatchSegmentationRequest(BaseModel):
    customers: List[CustomerDataRequest]
    segmentation_method: str = Field(default="kmeans", regex="^(kmeans|dbscan|hierarchical)$")
    prediction_model: str = Field(default="ensemble", regex="^(ensemble|random_forest|gradient_boosting|linear_regression|neural_network)$")

class BatchSegmentationResponse(BaseModel):
    results: List[SegmentationResponse]
    total_customers: int
    segment_distribution: Dict[str, int]
    average_prediction: float
    timestamp: datetime

# Initialize the system
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
    "chroma_db_path": "./retail_analytics_chroma_db",
    "llm_model_name": "gpt-4",
    "num_segments": 5,
    "input_dim": 25,
    "hidden_dims": [64, 32, 16],
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 5,
    "kmeans_model_path": "./models/retail_kmeans.pkl",
    "dbscan_model_path": "./models/retail_dbscan.pkl",
    "hierarchical_model_path": "./models/retail_hierarchical.pkl",
    "random_forest_model_path": "./models/retail_random_forest.pkl",
    "gradient_boosting_model_path": "./models/retail_gradient_boosting.pkl",
    "linear_regression_model_path": "./models/retail_linear_regression.pkl",
    "neural_network_weights_path": "./models/retail_neural_network.pth",
    "market_research_path": "./data/market_research",
    "behavior_studies_path": "./data/behavior_studies",
    "marketing_strategies_path": "./data/marketing_strategies",
    "industry_trends_path": "./data/industry_trends",
    "competitive_analysis_path": "./data/competitive_analysis",
    "product_catalog_path": "./data/product_catalog",
    "seasonal_patterns_path": "./data/seasonal_patterns",
    "pricing_strategies_path": "./data/pricing_strategies"
}

retail_analytics_system = RetailAnalyticsSystem(config)

@app.post("/segment_customer", response_model=SegmentationResponse)
async def segment_customer(request: CustomerDataRequest):
    """Segment customer and predict purchase behavior"""
    try:
        # Convert request to CustomerData
        customer = CustomerData(
            customer_id=request.customer_id,
            age=request.age,
            gender=request.gender,
            income=request.income,
            location=request.location,
            purchase_history=request.purchase_history,
            browsing_history=request.browsing_history,
            preferences=request.preferences,
            loyalty_score=request.loyalty_score,
            total_spent=request.total_spent,
            avg_order_value=request.avg_order_value,
            frequency_score=request.frequency_score,
            recency_score=request.recency_score,
            category_preferences=request.category_preferences,
            seasonal_patterns=request.seasonal_patterns
        )
        
        # Analyze customer
        result = retail_analytics_system.analyze_customer(customer)
        
        # Convert to response format
        response = SegmentationResponse(
            customer_id=result.customer_id,
            segment_id=result.segment_id,
            segment_name=result.segment_name,
            segment_description=result.segment_description,
            segment_characteristics=result.segment_characteristics,
            purchase_prediction=result.purchase_prediction,
            next_purchase_date=result.next_purchase_date,
            recommended_products=result.recommended_products,
            marketing_strategy=result.marketing_strategy,
            confidence_score=result.confidence_score,
            timestamp=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in segment customer endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_segment_customers", response_model=BatchSegmentationResponse)
async def batch_segment_customers(request: BatchSegmentationRequest):
    """Segment multiple customers"""
    try:
        results = []
        total_prediction = 0.0
        segment_counts = {}
        
        # Process each customer
        for customer_request in request.customers:
            # Convert to CustomerData
            customer = CustomerData(
                customer_id=customer_request.customer_id,
                age=customer_request.age,
                gender=customer_request.gender,
                income=customer_request.income,
                location=customer_request.location,
                purchase_history=customer_request.purchase_history,
                browsing_history=customer_request.browsing_history,
                preferences=customer_request.preferences,
                loyalty_score=customer_request.loyalty_score,
                total_spent=customer_request.total_spent,
                avg_order_value=customer_request.avg_order_value,
                frequency_score=customer_request.frequency_score,
                recency_score=customer_request.recency_score,
                category_preferences=customer_request.category_preferences,
                seasonal_patterns=customer_request.seasonal_patterns
            )
            
            # Analyze customer
            result = retail_analytics_system.analyze_customer(customer)
            
            # Convert to response format
            response = SegmentationResponse(
                customer_id=result.customer_id,
                segment_id=result.segment_id,
                segment_name=result.segment_name,
                segment_description=result.segment_description,
                segment_characteristics=result.segment_characteristics,
                purchase_prediction=result.purchase_prediction,
                next_purchase_date=result.next_purchase_date,
                recommended_products=result.recommended_products,
                marketing_strategy=result.marketing_strategy,
                confidence_score=result.confidence_score,
                timestamp=datetime.now()
            )
            
            results.append(response)
            total_prediction += result.purchase_prediction
            
            # Count segments
            segment_name = result.segment_name
            segment_counts[segment_name] = segment_counts.get(segment_name, 0) + 1
        
        # Calculate average prediction
        average_prediction = total_prediction / len(results) if results else 0.0
        
        return BatchSegmentationResponse(
            results=results,
            total_customers=len(results),
            segment_distribution=segment_counts,
            average_prediction=average_prediction,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in batch segment customers endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/segments/info")
async def get_segments_info():
    """Get information about customer segments"""
    try:
        return {
            "segments": retail_analytics_system.segment_descriptions,
            "total_segments": len(retail_analytics_system.segment_descriptions),
            "segmentation_methods": list(retail_analytics_system.segmentation_models.keys()),
            "prediction_models": list(retail_analytics_system.prediction_models.keys())
        }
    except Exception as e:
        logger.error(f"Error getting segments info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service": "Retail Customer Segmentation API",
        "segmentation_models_loaded": len(retail_analytics_system.segmentation_models),
        "prediction_models_loaded": len(retail_analytics_system.prediction_models),
        "rag_system_ready": retail_analytics_system.rag_system.retriever is not None,
        "device": str(retail_analytics_system.device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007) 