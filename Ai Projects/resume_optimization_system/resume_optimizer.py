"""
Intelligent Resume Optimization System Using LLMs and CrewAI with Enhanced RAG Integration
A comprehensive system for optimizing resumes with multi-agent collaboration and job market intelligence
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

# Core AI/ML Libraries
import openai
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ChromaDB, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# CrewAI Framework
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Document Processing
import PyPDF2
from docx import Document
import pandas as pd
import numpy as np

# Web Scraping and API Integration
import requests
from bs4 import BeautifulSoup
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Web Framework
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
from config import (
    OPENAI_API_KEY,
    CHROMA_DB_PATH,
    JOB_MARKET_DATA_PATH,
    RESUME_TEMPLATES_PATH,
    RAG_PARAMS,
    KNOWLEDGE_SOURCES
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Resume Optimization System API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class ResumeData:
    """Data class for resume information with enhanced metadata"""
    content: str
    format: str  # pdf, docx, txt
    sections: Dict[str, str]
    metadata: Dict[str, Any]
    original_file_path: str
    job_context: Dict[str, Any]
    market_insights: Dict[str, Any]

@dataclass
class JobRequirement:
    """Data class for job requirements with enhanced market data"""
    title: str
    company: str
    requirements: List[str]
    skills: List[str]
    experience_level: str
    industry: str
    market_trends: Dict[str, Any]
    salary_range: Dict[str, Any]
    location: str

@dataclass
class OptimizationResult:
    """Data class for optimization results with enhanced insights"""
    original_resume: ResumeData
    optimized_resume: str
    improvements: List[str]
    keyword_suggestions: List[str]
    ats_score: float
    recommendations: List[str]
    sources: List[str]
    market_insights: Dict[str, Any]
    industry_trends: Dict[str, Any]

class JobMarketRAGSystem:
    """Enhanced RAG system for accessing job market data and trends with market intelligence"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.bm25_retriever = None
        self.knowledge_base = {}
        self.retrieval_cache = {}
        self.market_databases = {}
        self.initialize_knowledge_base()
        self.setup_retrievers()
    
    def initialize_knowledge_base(self):
        """Initialize job market knowledge base with enhanced RAG capabilities"""
        try:
            # Initialize vector store
            self.vector_store = ChromaDB(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings
            )
            
            # Load comprehensive job market knowledge sources
            self.load_job_market_sources()
            self.load_market_databases()
            logger.info("Job market knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            raise
    
    def setup_retrievers(self):
        """Setup multiple retrieval strategies for enhanced job market information retrieval"""
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
            
            logger.info("Job market retrievers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
    
    def load_job_market_sources(self):
        """Load comprehensive job market knowledge sources from various repositories"""
        try:
            # Load from local job market knowledge base
            if os.path.exists(JOB_MARKET_DATA_PATH):
                self.load_job_market_documents("local")
            
            # Load from job posting databases
            self.load_job_postings()
            
            # Load from industry reports
            self.load_industry_reports()
            
            # Load from ATS guidelines
            self.load_ats_guidelines()
            
            # Load from salary data
            self.load_salary_data()
            
            # Load from career development resources
            self.load_career_resources()
            
            logger.info(f"Loaded {len(self.knowledge_base)} job market knowledge sources")
            
        except Exception as e:
            logger.error(f"Error loading job market sources: {e}")
    
    def load_market_databases(self):
        """Load specialized market databases for job analysis"""
        try:
            # Load LinkedIn job data
            self.load_linkedin_job_data()
            
            # Load Glassdoor data
            self.load_glassdoor_data()
            
            # Load Indeed job data
            self.load_indeed_data()
            
            # Load industry trend data
            self.load_industry_trends()
            
            logger.info(f"Loaded {len(self.market_databases)} market databases")
            
        except Exception as e:
            logger.error(f"Error loading market databases: {e}")
    
    def load_job_market_documents(self, source_type: str) -> List[str]:
        """Load job market documents from specified source"""
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
    
    def load_job_postings(self) -> List[str]:
        """Load job posting data from various sources"""
        try:
            # This would integrate with job posting APIs
            # For now, we'll simulate with sample data
            job_postings = [
                {
                    "title": "Software Engineer",
                    "company": "Tech Corp",
                    "requirements": ["Python", "JavaScript", "React", "AWS"],
                    "experience": "3-5 years",
                    "industry": "Technology",
                    "location": "San Francisco, CA"
                },
                {
                    "title": "Data Scientist",
                    "company": "Analytics Inc",
                    "requirements": ["Python", "Machine Learning", "SQL", "Statistics"],
                    "experience": "2-4 years",
                    "industry": "Technology",
                    "location": "New York, NY"
                }
            ]
            
            for posting in job_postings:
                self.knowledge_base[f"job_{posting['title']}_{posting['company']}"] = {
                    "content": f"Job: {posting['title']}\nCompany: {posting['company']}\nRequirements: {', '.join(posting['requirements'])}",
                    "metadata": posting,
                    "type": "job_posting"
                }
                
        except Exception as e:
            logger.error(f"Error loading job postings: {e}")
    
    def load_industry_reports(self) -> List[str]:
        """Load industry reports and market analysis"""
        try:
            # Load industry reports
            reports = [
                {
                    "industry": "Technology",
                    "title": "2024 Tech Job Market Trends",
                    "content": "AI and machine learning skills are in high demand...",
                    "trends": ["AI/ML", "Cloud Computing", "Cybersecurity"],
                    "year": 2024
                },
                {
                    "industry": "Healthcare",
                    "title": "Healthcare Job Market Analysis",
                    "content": "Telemedicine and digital health are growing rapidly...",
                    "trends": ["Telemedicine", "Digital Health", "Data Analytics"],
                    "year": 2024
                }
            ]
            
            for report in reports:
                self.knowledge_base[f"report_{report['industry']}_{report['title']}"] = {
                    "content": f"Industry: {report['industry']}\nTitle: {report['title']}\nContent: {report['content']}",
                    "metadata": report,
                    "type": "industry_report"
                }
                
        except Exception as e:
            logger.error(f"Error loading industry reports: {e}")
    
    def load_ats_guidelines(self) -> List[str]:
        """Load ATS (Applicant Tracking System) guidelines"""
        try:
            # Load ATS guidelines
            ats_guidelines = [
                {
                    "system": "Workday",
                    "guidelines": "Use standard section headers, avoid graphics, use keywords",
                    "best_practices": ["Clear formatting", "Keyword optimization", "Standard fonts"]
                },
                {
                    "system": "BambooHR",
                    "guidelines": "Simple formatting, bullet points, relevant keywords",
                    "best_practices": ["Simple layout", "Action verbs", "Quantified achievements"]
                }
            ]
            
            for guideline in ats_guidelines:
                self.knowledge_base[f"ats_{guideline['system']}"] = {
                    "content": f"System: {guideline['system']}\nGuidelines: {guideline['guidelines']}",
                    "metadata": guideline,
                    "type": "ats_guideline"
                }
                
        except Exception as e:
            logger.error(f"Error loading ATS guidelines: {e}")
    
    def load_salary_data(self):
        """Load salary and compensation data"""
        try:
            # Load salary data
            salary_data = [
                {
                    "title": "Software Engineer",
                    "location": "San Francisco",
                    "salary_range": {"min": 120000, "max": 180000},
                    "experience": "3-5 years",
                    "industry": "Technology"
                },
                {
                    "title": "Data Scientist",
                    "location": "New York",
                    "salary_range": {"min": 100000, "max": 150000},
                    "experience": "2-4 years",
                    "industry": "Technology"
                }
            ]
            
            for salary in salary_data:
                self.knowledge_base[f"salary_{salary['title']}_{salary['location']}"] = {
                    "content": f"Title: {salary['title']}\nLocation: {salary['location']}\nSalary: ${salary['salary_range']['min']}-{salary['salary_range']['max']}",
                    "metadata": salary,
                    "type": "salary_data"
                }
                
        except Exception as e:
            logger.error(f"Error loading salary data: {e}")
    
    def load_career_resources(self):
        """Load career development and resume writing resources"""
        try:
            # Load career resources
            career_resources = [
                {
                    "topic": "Resume Writing",
                    "content": "Best practices for writing effective resumes",
                    "tips": ["Use action verbs", "Quantify achievements", "Tailor to job"],
                    "source": "Career Development Guide"
                },
                {
                    "topic": "Interview Preparation",
                    "content": "How to prepare for job interviews",
                    "tips": ["Research company", "Practice responses", "Prepare questions"],
                    "source": "Interview Guide"
                }
            ]
            
            for resource in career_resources:
                self.knowledge_base[f"career_{resource['topic']}"] = {
                    "content": f"Topic: {resource['topic']}\nContent: {resource['content']}",
                    "metadata": resource,
                    "type": "career_resource"
                }
                
        except Exception as e:
            logger.error(f"Error loading career resources: {e}")
    
    def load_linkedin_job_data(self):
        """Load LinkedIn job market data"""
        try:
            # Load LinkedIn data (simulated)
            linkedin_data = {
                "trending_skills": ["Python", "Machine Learning", "Cloud Computing"],
                "job_growth": {"Technology": 15, "Healthcare": 12, "Finance": 8},
                "top_companies": ["Google", "Microsoft", "Amazon"]
            }
            
            self.market_databases["linkedin"] = linkedin_data
            
        except Exception as e:
            logger.error(f"Error loading LinkedIn job data: {e}")
    
    def load_glassdoor_data(self):
        """Load Glassdoor salary and company data"""
        try:
            # Load Glassdoor data (simulated)
            glassdoor_data = {
                "company_ratings": {"Google": 4.5, "Microsoft": 4.3, "Amazon": 3.8},
                "salary_insights": {"Software Engineer": 130000, "Data Scientist": 120000},
                "interview_difficulty": {"Google": "Hard", "Microsoft": "Medium", "Amazon": "Medium"}
            }
            
            self.market_databases["glassdoor"] = glassdoor_data
            
        except Exception as e:
            logger.error(f"Error loading Glassdoor data: {e}")
    
    def load_indeed_data(self):
        """Load Indeed job market data"""
        try:
            # Load Indeed data (simulated)
            indeed_data = {
                "job_postings": 1000000,
                "trending_jobs": ["Remote Work", "AI Engineer", "DevOps Engineer"],
                "salary_trends": {"increasing": ["AI/ML", "Cybersecurity"], "stable": ["Marketing", "Sales"]}
            }
            
            self.market_databases["indeed"] = indeed_data
            
        except Exception as e:
            logger.error(f"Error loading Indeed data: {e}")
    
    def load_industry_trends(self):
        """Load industry trend data"""
        try:
            # Load industry trends (simulated)
            industry_trends = {
                "technology": {
                    "growth_rate": 15,
                    "hot_skills": ["AI/ML", "Cloud", "Cybersecurity"],
                    "emerging_roles": ["Prompt Engineer", "AI Ethics Specialist"]
                },
                "healthcare": {
                    "growth_rate": 12,
                    "hot_skills": ["Telemedicine", "Data Analytics", "Digital Health"],
                    "emerging_roles": ["Health Data Analyst", "Telehealth Coordinator"]
                }
            }
            
            self.market_databases["industry_trends"] = industry_trends
            
        except Exception as e:
            logger.error(f"Error loading industry trends: {e}")
    
    def retrieve_job_market_info(self, query: str, job_context: Dict[str, Any] = None, 
                                top_k: int = 5, use_ensemble: bool = True) -> List[Dict]:
        """Enhanced retrieval with job context awareness and multiple strategies"""
        try:
            # Check cache first
            cache_key = f"{query}_{str(job_context)}_{top_k}_{use_ensemble}"
            if cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            # Enhance query with job context
            if job_context:
                enhanced_query = self.enhance_query_with_job_context(query, job_context)
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
                    "relevance_score": self.calculate_job_market_relevance_score(enhanced_query, doc.page_content, job_context),
                    "evidence_level": doc.metadata.get("evidence_level", "unknown")
                }
                retrieved_info.append(info)
            
            # Sort by relevance score
            retrieved_info.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Cache results
            self.retrieval_cache[cache_key] = retrieved_info[:top_k]
            
            logger.info(f"Retrieved {len(retrieved_info)} relevant documents for job market query: {query[:50]}...")
            return retrieved_info[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving job market info: {e}")
            return []
    
    def enhance_query_with_job_context(self, query: str, job_context: Dict[str, Any]) -> str:
        """Enhance query with job-specific context"""
        try:
            enhanced_parts = [query]
            
            # Add job title context
            if 'title' in job_context:
                enhanced_parts.append(f"job title {job_context['title']}")
            if 'industry' in job_context:
                enhanced_parts.append(f"industry {job_context['industry']}")
            
            # Add location context
            if 'location' in job_context:
                enhanced_parts.append(f"location {job_context['location']}")
            
            # Add experience level context
            if 'experience_level' in job_context:
                enhanced_parts.append(f"experience {job_context['experience_level']}")
            
            return " ".join(enhanced_parts)
            
        except Exception as e:
            logger.error(f"Error enhancing query with job context: {e}")
            return query
    
    def calculate_job_market_relevance_score(self, query: str, content: str, job_context: Dict[str, Any] = None) -> float:
        """Calculate relevance score for job market content with job context"""
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
            
            # Boost score for job market-related terms
            job_market_terms = [
                "job", "career", "employment", "salary", "skills", "experience",
                "industry", "market", "trend", "requirement", "qualification",
                "resume", "interview", "application", "hiring", "recruitment"
            ]
            job_market_boost = sum(1 for term in job_market_terms if term in content.lower())
            relevance += job_market_boost * 0.1
            
            # Boost for job-specific context
            if job_context:
                # Boost for title-specific content
                if 'title' in job_context:
                    title = job_context['title']
                    if title.lower() in content.lower():
                        relevance += 0.15
                
                # Boost for industry-specific content
                if 'industry' in job_context:
                    industry = job_context['industry']
                    if industry.lower() in content.lower():
                        relevance += 0.15
            
            # Boost for evidence-based content
            evidence_indicators = ["study", "research", "report", "analysis", "data"]
            evidence_boost = sum(1 for indicator in evidence_indicators if indicator in content.lower())
            relevance += evidence_boost * 0.05
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating job market relevance score: {e}")
            return 0.0

class ResumeParser:
    """Parse resumes from different formats"""
    
    def __init__(self):
        self.supported_formats = ['pdf', 'docx', 'txt']
    
    def parse_resume(self, file_path: str, file_format: str) -> ResumeData:
        """Parse resume from file"""
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_format}")
        
        if file_format == 'pdf':
            return self.parse_pdf(file_path)
        elif file_format == 'docx':
            return self.parse_docx(file_path)
        elif file_format == 'txt':
            return self.parse_txt(file_path)
    
    def parse_pdf(self, file_path: str) -> ResumeData:
        """Parse PDF resume"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
            
            sections = self.extract_sections(content)
            metadata = self.extract_metadata(content)
            
            return ResumeData(
                content=content,
                format='pdf',
                sections=sections,
                metadata=metadata,
                original_file_path=file_path,
                job_context={}, # Placeholder, will be updated by RAG
                market_insights={} # Placeholder, will be updated by RAG
            )
            
        except Exception as e:
            raise Exception(f"Error parsing PDF: {e}")
    
    def parse_docx(self, file_path: str) -> ResumeData:
        """Parse DOCX resume"""
        try:
            doc = Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            sections = self.extract_sections(content)
            metadata = self.extract_metadata(content)
            
            return ResumeData(
                content=content,
                format='docx',
                sections=sections,
                metadata=metadata,
                original_file_path=file_path,
                job_context={}, # Placeholder, will be updated by RAG
                market_insights={} # Placeholder, will be updated by RAG
            )
            
        except Exception as e:
            raise Exception(f"Error parsing DOCX: {e}")
    
    def parse_txt(self, file_path: str) -> ResumeData:
        """Parse TXT resume"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            sections = self.extract_sections(content)
            metadata = self.extract_metadata(content)
            
            return ResumeData(
                content=content,
                format='txt',
                sections=sections,
                metadata=metadata,
                original_file_path=file_path,
                job_context={}, # Placeholder, will be updated by RAG
                market_insights={} # Placeholder, will be updated by RAG
            )
            
        except Exception as e:
            raise Exception(f"Error parsing TXT: {e}")
    
    def extract_sections(self, content: str) -> Dict[str, str]:
        """Extract resume sections"""
        sections = {}
        
        # Common section headers
        section_headers = [
            'experience', 'work experience', 'employment history',
            'education', 'academic background', 'qualifications',
            'skills', 'technical skills', 'competencies',
            'projects', 'achievements', 'accomplishments',
            'summary', 'objective', 'profile'
        ]
        
        lines = content.split('\n')
        current_section = 'general'
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            is_header = any(header in line_lower for header in section_headers)
            
            if is_header and current_content:
                sections[current_section] = '\n'.join(current_content)
                current_section = line_lower
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from resume"""
        metadata = {}
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        if emails:
            metadata['email'] = emails[0]
        
        # Extract phone
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, content)
        if phones:
            metadata['phone'] = phones[0]
        
        # Extract name (first few lines)
        lines = content.split('\n')[:5]
        for line in lines:
            if line.strip() and not any(keyword in line.lower() for keyword in ['email', 'phone', 'address']):
                metadata['name'] = line.strip()
                break
        
        return metadata

class ResumeOptimizationCrew:
    """CrewAI-based resume optimization system"""
    
    def __init__(self, rag_system: JobMarketRAGSystem):
        self.rag_system = rag_system
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        self.agents = self.create_agents()
    
    def create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents for resume optimization"""
        
        # Resume Parser Agent
        parser_agent = Agent(
            role='Resume Parser',
            goal='Extract and analyze resume content comprehensively',
            backstory="""You are an expert at analyzing resumes and extracting 
            key information including skills, experience, education, and achievements. 
            You can identify strengths and weaknesses in resume content.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Content Analysis Agent
        content_agent = Agent(
            role='Content Analyst',
            goal='Analyze resume content for clarity, impact, and effectiveness',
            backstory="""You are a professional resume writer and career coach 
            with years of experience helping job seekers create compelling resumes. 
            You understand what makes resumes stand out and how to improve them.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Market Intelligence Agent
        market_agent = Agent(
            role='Market Intelligence Specialist',
            goal='Provide current job market insights and trends',
            backstory="""You are a job market analyst who stays current with 
            industry trends, in-demand skills, and job market dynamics. 
            You understand what employers are looking for in different industries.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Keyword Optimization Agent
        keyword_agent = Agent(
            role='ATS Optimization Specialist',
            goal='Optimize resume for applicant tracking systems',
            backstory="""You are an expert in ATS optimization and keyword 
            matching. You understand how applicant tracking systems work and 
            how to ensure resumes pass through them successfully.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Formatting Agent
        formatting_agent = Agent(
            role='Resume Formatting Expert',
            goal='Optimize resume layout and visual presentation',
            backstory="""You are a professional resume designer who understands 
            the importance of clean, professional formatting. You know how to 
            make resumes visually appealing while maintaining ATS compatibility.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        return {
            'parser': parser_agent,
            'content': content_agent,
            'market': market_agent,
            'keyword': keyword_agent,
            'formatting': formatting_agent
        }
    
    async def optimize_resume(self, resume_data: ResumeData, 
                            job_requirements: Optional[JobRequirement] = None) -> OptimizationResult:
        """Optimize resume using CrewAI agents"""
        
        # Create tasks for each agent
        tasks = []
        
        # Task 1: Parse and analyze resume
        parse_task = Task(
            description=f"""
            Analyze the following resume content and extract key information:
            
            Resume Content:
            {resume_data.content}
            
            Provide a detailed analysis including:
            1. Key skills identified
            2. Experience summary
            3. Education and qualifications
            4. Strengths and areas for improvement
            5. Overall assessment
            """,
            agent=self.agents['parser'],
            expected_output="Detailed resume analysis report"
        )
        tasks.append(parse_task)
        
        # Task 2: Content analysis and improvement
        content_task = Task(
            description=f"""
            Based on the resume analysis, provide specific recommendations for:
            1. Improving clarity and impact
            2. Enhancing achievement descriptions
            3. Making content more compelling
            4. Addressing any gaps or weaknesses
            5. Strengthening the professional summary
            
            Consider the target job requirements if provided.
            """,
            agent=self.agents['content'],
            expected_output="Content improvement recommendations",
            context=[parse_task]
        )
        tasks.append(content_task)
        
        # Task 3: Market intelligence
        market_task = Task(
            description=f"""
            Provide current job market insights for:
            1. In-demand skills for the target role/industry
            2. Current salary trends
            3. Industry-specific requirements
            4. Emerging trends and technologies
            5. Competitive landscape analysis
            
            Use the latest market data and trends.
            """,
            agent=self.agents['market'],
            expected_output="Market intelligence report"
        )
        tasks.append(market_task)
        
        # Task 4: Keyword optimization
        keyword_task = Task(
            description=f"""
            Optimize the resume for ATS systems by:
            1. Identifying relevant keywords for the target role
            2. Suggesting keyword placement strategies
            3. Ensuring ATS compatibility
            4. Optimizing for specific job requirements
            5. Providing keyword density recommendations
            
            Focus on both human readability and ATS optimization.
            """,
            agent=self.agents['keyword'],
            expected_output="ATS optimization recommendations",
            context=[parse_task, market_task]
        )
        tasks.append(keyword_task)
        
        # Task 5: Formatting optimization
        formatting_task = Task(
            description=f"""
            Provide formatting recommendations for:
            1. Professional layout and structure
            2. Font and spacing optimization
            3. Section organization
            4. Visual hierarchy
            5. ATS-friendly formatting
            
            Ensure the resume is both visually appealing and ATS-compatible.
            """,
            agent=self.agents['formatting'],
            expected_output="Formatting optimization recommendations",
            context=[parse_task, content_task]
        )
        tasks.append(formatting_task)
        
        # Create and run the crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        
        # Execute the optimization process
        result = await crew.kickoff()
        
        # Process results and create optimization result
        return self.process_optimization_results(result, resume_data, job_requirements)
    
    def process_optimization_results(self, crew_result: str, resume_data: ResumeData, 
                                   job_requirements: Optional[JobRequirement]) -> OptimizationResult:
        """Process crew results and create optimization result"""
        
        # Extract recommendations from crew result
        recommendations = self.extract_recommendations(crew_result)
        
        # Generate optimized resume
        optimized_resume = self.generate_optimized_resume(resume_data, recommendations)
        
        # Calculate ATS score
        ats_score = self.calculate_ats_score(optimized_resume, job_requirements)
        
        # Generate keyword suggestions
        keyword_suggestions = self.generate_keyword_suggestions(job_requirements)
        
        # Identify improvements
        improvements = self.identify_improvements(resume_data, optimized_resume)
        
        # Update job_context and market_insights in resume_data
        resume_data.job_context = self.rag_system.retrieve_job_market_info(optimized_resume, job_requirements)
        resume_data.market_insights = self.rag_system.market_databases
        
        return OptimizationResult(
            original_resume=resume_data,
            optimized_resume=optimized_resume,
            improvements=improvements,
            keyword_suggestions=keyword_suggestions,
            ats_score=ats_score,
            recommendations=recommendations,
            market_insights=resume_data.market_insights,
            industry_trends=self.rag_system.market_databases["industry_trends"]
        )
    
    def extract_recommendations(self, crew_result: str) -> List[str]:
        """Extract recommendations from crew result"""
        # Parse the crew result to extract specific recommendations
        recommendations = []
        
        # Simple parsing - in a real system, this would be more sophisticated
        lines = crew_result.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'improve', 'enhance']):
                recommendations.append(line.strip())
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def generate_optimized_resume(self, resume_data: ResumeData, 
                                recommendations: List[str]) -> str:
        """Generate optimized resume based on recommendations"""
        
        # This would implement the actual resume optimization logic
        # For now, return a placeholder
        optimized_content = resume_data.content
        
        # Apply basic optimizations
        optimized_content = self.apply_basic_optimizations(optimized_content, recommendations)
        
        return optimized_content
    
    def apply_basic_optimizations(self, content: str, recommendations: List[str]) -> str:
        """Apply basic optimizations to resume content"""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Ensure proper bullet points
        content = re.sub(r'^\s*[-*]\s*', '• ', content, flags=re.MULTILINE)
        
        # Add action verbs where appropriate
        action_verbs = ['developed', 'implemented', 'managed', 'created', 'improved']
        # This is a simplified version - in practice, you'd use NLP for this
        
        return content
    
    def calculate_ats_score(self, optimized_resume: str, 
                          job_requirements: Optional[JobRequirement]) -> float:
        """Calculate ATS compatibility score"""
        
        if not job_requirements:
            return 0.7  # Default score
        
        # Count keyword matches
        resume_lower = optimized_resume.lower()
        requirements_lower = ' '.join(job_requirements.requirements).lower()
        skills_lower = ' '.join(job_requirements.skills).lower()
        
        # Calculate keyword match percentage
        requirement_words = set(requirements_lower.split())
        skill_words = set(skills_lower.split())
        
        resume_words = set(resume_lower.split())
        
        requirement_match = len(requirement_words.intersection(resume_words)) / len(requirement_words)
        skill_match = len(skill_words.intersection(resume_words)) / len(skill_words)
        
        # Weighted score
        ats_score = (requirement_match * 0.6) + (skill_match * 0.4)
        
        return min(ats_score, 1.0)
    
    def generate_keyword_suggestions(self, job_requirements: Optional[JobRequirement]) -> List[str]:
        """Generate keyword suggestions for optimization"""
        
        if not job_requirements:
            return ["leadership", "project management", "communication", "problem solving"]
        
        # Combine requirements and skills
        all_keywords = job_requirements.requirements + job_requirements.skills
        
        # Remove common words and duplicates
        filtered_keywords = []
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for keyword in all_keywords:
            if keyword.lower() not in common_words and keyword not in filtered_keywords:
                filtered_keywords.append(keyword)
        
        return filtered_keywords[:15]  # Return top 15 keywords
    
    def identify_improvements(self, original_resume: ResumeData, 
                            optimized_resume: str) -> List[str]:
        """Identify specific improvements made"""
        
        improvements = []
        
        # Compare content length
        if len(optimized_resume) > len(original_resume.content):
            improvements.append("Enhanced content with more detailed descriptions")
        
        # Check for action verbs
        action_verbs = ['developed', 'implemented', 'managed', 'created', 'improved', 'led', 'designed']
        original_verb_count = sum(1 for verb in action_verbs if verb in original_resume.content.lower())
        optimized_verb_count = sum(1 for verb in action_verbs if verb in optimized_resume.lower())
        
        if optimized_verb_count > original_verb_count:
            improvements.append("Added more action verbs for stronger impact")
        
        # Check for bullet points
        original_bullets = original_resume.content.count('•')
        optimized_bullets = optimized_resume.count('•')
        
        if optimized_bullets > original_bullets:
            improvements.append("Improved formatting with better bullet point usage")
        
        return improvements
    
    def get_sources(self) -> List[str]:
        """Get sources used for optimization"""
        return [
            "Job market analysis data",
            "ATS optimization guidelines",
            "Resume writing best practices",
            "Industry trend reports"
        ]

class ResumeOptimizationSystem:
    """Main resume optimization system"""
    
    def __init__(self):
        self.rag_system = JobMarketRAGSystem()
        self.parser = ResumeParser()
        self.crew = ResumeOptimizationCrew(self.rag_system)
    
    async def optimize_resume(self, resume_file_path: str, 
                            job_title: str = None, 
                            company: str = None) -> OptimizationResult:
        """Main method to optimize a resume"""
        
        try:
            # Parse resume
            file_format = resume_file_path.split('.')[-1].lower()
            resume_data = self.parser.parse_resume(resume_file_path, file_format)
            
            # Create job requirements if provided
            job_requirements = None
            if job_title and company:
                job_requirements = JobRequirement(
                    title=job_title,
                    company=company,
                    requirements=self.get_job_requirements(job_title),
                    skills=self.get_job_skills(job_title),
                    experience_level="mid-level",
                    industry="technology",
                    market_trends={}, # Placeholder, will be updated by RAG
                    salary_range={}, # Placeholder, will be updated by RAG
                    location="San Francisco" # Placeholder
                )
            
            # Optimize resume using CrewAI
            result = await self.crew.optimize_resume(resume_data, job_requirements)
            
            return result
            
        except Exception as e:
            raise Exception(f"Error optimizing resume: {e}")
    
    def get_job_requirements(self, job_title: str) -> List[str]:
        """Get job requirements for a specific title"""
        # This would typically query a job database or API
        # For now, return sample requirements
        requirements_map = {
            "software engineer": [
                "Proficiency in programming languages",
                "Experience with software development",
                "Knowledge of algorithms and data structures",
                "Familiarity with version control systems"
            ],
            "data scientist": [
                "Experience with machine learning",
                "Proficiency in Python and R",
                "Knowledge of statistical analysis",
                "Experience with data visualization"
            ],
            "product manager": [
                "Experience in product strategy",
                "Strong analytical skills",
                "Excellent communication abilities",
                "Experience with agile methodologies"
            ]
        }
        
        return requirements_map.get(job_title.lower(), ["Strong communication skills", "Problem-solving abilities"])
    
    def get_job_skills(self, job_title: str) -> List[str]:
        """Get required skills for a specific job title"""
        # This would typically query a skills database
        skills_map = {
            "software engineer": ["Python", "Java", "JavaScript", "Git", "SQL"],
            "data scientist": ["Python", "R", "SQL", "Machine Learning", "Statistics"],
            "product manager": ["Product Strategy", "Analytics", "User Research", "Agile", "SQL"]
        }
        
        return skills_map.get(job_title.lower(), ["Communication", "Leadership", "Problem Solving"])

# Pydantic models for API
class ResumeOptimizationRequest(BaseModel):
    resume_file_path: str
    job_title: Optional[str] = None
    company: Optional[str] = None
    include_ats_score: bool = True

class ResumeOptimizationResponse(BaseModel):
    optimized_resume: str
    improvements: List[str]
    keyword_suggestions: List[str]
    ats_score: float
    recommendations: List[str]
    sources: List[str]
    market_insights: Dict[str, Any]
    industry_trends: Dict[str, Any]

# Initialize system
resume_optimizer = ResumeOptimizationSystem()

@app.post("/optimize", response_model=ResumeOptimizationResponse)
async def optimize_resume_endpoint(request: ResumeOptimizationRequest):
    """Optimize resume endpoint"""
    try:
        result = await resume_optimizer.optimize_resume(
            request.resume_file_path,
            request.job_title,
            request.company
        )
        
        return ResumeOptimizationResponse(
            optimized_resume=result.optimized_resume,
            improvements=result.improvements,
            keyword_suggestions=result.keyword_suggestions,
            ats_score=result.ats_score,
            recommendations=result.recommendations,
            sources=result.sources,
            market_insights=result.market_insights,
            industry_trends=result.industry_trends
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_and_optimize")
async def upload_and_optimize_resume(
    file: UploadFile = File(...),
    job_title: str = None,
    company: str = None
):
    """Upload and optimize resume endpoint"""
    try:
        # Save uploaded file
        file_path = f"./uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Optimize resume
        result = await resume_optimizer.optimize_resume(
            file_path,
            job_title,
            company
        )
        
        return {
            "message": "Resume optimized successfully",
            "optimized_resume": result.optimized_resume,
            "improvements": result.improvements,
            "ats_score": result.ats_score,
            "recommendations": result.recommendations,
            "market_insights": result.market_insights,
            "industry_trends": result.industry_trends
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 