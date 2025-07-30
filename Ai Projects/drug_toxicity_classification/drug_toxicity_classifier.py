"""
Drug Toxicity Classification System with Enhanced RAG Integration
A comprehensive system for classifying drug compounds based on toxicity levels with advanced research literature integration
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Core ML Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Chemical Informatics Libraries
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
import openchem

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch_geometric
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv, global_mean_pool

# Enhanced RAG and Vector Search
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import ChromaDB, Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI

# Web Framework
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
from config import (
    OPENAI_API_KEY, 
    CHROMA_DB_PATH,
    TOXICITY_DATA_PATH,
    RESEARCH_LITERATURE_PATH,
    RAG_PARAMS,
    KNOWLEDGE_SOURCES
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Drug Toxicity Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class DrugCompound:
    """Data class for drug compound information"""
    smiles: str
    name: str
    molecular_weight: float
    descriptors: Dict[str, float]
    toxicity_data: Dict[str, Any]
    source: str
    research_context: Dict[str, Any]

@dataclass
class ToxicityPrediction:
    """Data class for toxicity prediction results"""
    compound_name: str
    smiles: str
    predicted_toxicity_class: str
    confidence_score: float
    toxicity_probabilities: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    sources: List[str]
    research_evidence: Dict[str, Any]
    molecular_insights: Dict[str, Any]

class ToxicityRAGSystem:
    """Enhanced RAG system for accessing toxicity research literature and guidelines with advanced retrieval"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.bm25_retriever = None
        self.knowledge_base = {}
        self.retrieval_cache = {}
        self.research_databases = {}
        self.initialize_knowledge_base()
        self.setup_retrievers()
    
    def initialize_knowledge_base(self):
        """Initialize toxicity knowledge base with enhanced RAG capabilities"""
        try:
            # Initialize vector store
            self.vector_store = ChromaDB(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings
            )
            
            # Load comprehensive toxicity knowledge sources
            self.load_toxicity_sources()
            self.load_research_databases()
            logger.info("Toxicity knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            raise
    
    def setup_retrievers(self):
        """Setup multiple retrieval strategies for enhanced toxicity information retrieval"""
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
            
            logger.info("Toxicity retrievers setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
    
    def load_toxicity_sources(self):
        """Load comprehensive toxicity knowledge sources from various repositories"""
        try:
            # Load from local toxicity knowledge base
            if os.path.exists(TOXICITY_DATA_PATH):
                self.load_toxicity_documents("local")
            
            # Load from PubMed toxicity articles
            self.load_pubmed_toxicity_articles()
            
            # Load from regulatory guidelines
            self.load_regulatory_guidelines()
            
            # Load from drug toxicity databases
            self.load_drug_toxicity_database()
            
            # Load from chemical safety databases
            self.load_chemical_safety_data()
            
            # Load from clinical trial data
            self.load_clinical_trial_data()
            
            logger.info(f"Loaded {len(self.knowledge_base)} toxicity knowledge sources")
            
        except Exception as e:
            logger.error(f"Error loading toxicity sources: {e}")
    
    def load_research_databases(self):
        """Load specialized research databases for toxicity analysis"""
        try:
            # Load ToxCast database
            self.load_toxcast_database()
            
            # Load ChEMBL database
            self.load_chembl_database()
            
            # Load PubChem toxicity data
            self.load_pubchem_toxicity()
            
            # Load FDA drug safety data
            self.load_fda_safety_data()
            
            logger.info(f"Loaded {len(self.research_databases)} research databases")
            
        except Exception as e:
            logger.error(f"Error loading research databases: {e}")
    
    def load_toxicity_documents(self, source_type: str) -> List[str]:
        """Load toxicity documents from specified source"""
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
    
    def load_pubmed_toxicity_articles(self) -> List[str]:
        """Load toxicity-related articles from PubMed API"""
        try:
            # This would integrate with PubMed API for toxicity research
            # For now, we'll simulate with sample data
            sample_articles = [
                {
                    "title": "Toxicity prediction of drug compounds using machine learning",
                    "abstract": "This study presents a comprehensive analysis of drug toxicity prediction...",
                    "authors": ["Dr. Chen", "Dr. Wang"],
                    "journal": "Journal of Medicinal Chemistry",
                    "year": 2024,
                    "pmid": "12345678"
                },
                {
                    "title": "Molecular mechanisms of drug-induced liver injury",
                    "abstract": "Drug-induced liver injury (DILI) is a major concern in drug development...",
                    "authors": ["Dr. Johnson", "Dr. Smith"],
                    "journal": "Toxicological Sciences",
                    "year": 2024,
                    "pmid": "87654321"
                }
            ]
            
            for article in sample_articles:
                self.knowledge_base[f"pubmed_{article['pmid']}"] = {
                    "content": f"{article['title']}\n{article['abstract']}",
                    "metadata": article,
                    "type": "pubmed"
                }
                
        except Exception as e:
            logger.error(f"Error loading PubMed articles: {e}")
    
    def load_regulatory_guidelines(self) -> List[str]:
        """Load regulatory guidelines for drug toxicity assessment"""
        try:
            # Load from regulatory agencies
            guidelines = [
                {
                    "title": "ICH S2(R1) Genotoxicity Testing Guidelines",
                    "content": "Guidelines for genotoxicity testing of pharmaceuticals...",
                    "source": "ICH",
                    "year": 2024
                },
                {
                    "title": "FDA Guidance on Drug-Induced Liver Injury",
                    "content": "Guidance for industry on assessing drug-induced liver injury...",
                    "source": "FDA",
                    "year": 2024
                }
            ]
            
            for guideline in guidelines:
                self.knowledge_base[f"guideline_{guideline['source']}_{guideline['title']}"] = {
                    "content": guideline['content'],
                    "metadata": guideline,
                    "type": "guideline"
                }
                
        except Exception as e:
            logger.error(f"Error loading regulatory guidelines: {e}")
    
    def load_drug_toxicity_database(self) -> List[str]:
        """Load comprehensive drug toxicity database"""
        try:
            # Load from drug toxicity databases
            toxicity_data = [
                {
                    "compound": "Acetaminophen",
                    "toxicity_class": "Low",
                    "mechanism": "Metabolic activation to NAPQI",
                    "target_organs": ["Liver"],
                    "dose_response": "Dose-dependent hepatotoxicity"
                },
                {
                    "compound": "Cisplatin",
                    "toxicity_class": "High",
                    "mechanism": "DNA cross-linking",
                    "target_organs": ["Kidney", "Nervous system"],
                    "dose_response": "Nephrotoxicity at therapeutic doses"
                }
            ]
            
            for compound in toxicity_data:
                self.knowledge_base[f"toxicity_{compound['compound']}"] = {
                    "content": f"Compound: {compound['compound']}\nToxicity Class: {compound['toxicity_class']}\nMechanism: {compound['mechanism']}",
                    "metadata": compound,
                    "type": "toxicity_data"
                }
                
        except Exception as e:
            logger.error(f"Error loading drug toxicity database: {e}")
    
    def load_chemical_safety_data(self):
        """Load chemical safety and hazard information"""
        try:
            # Load from chemical safety databases
            safety_data = [
                {
                    "compound": "Methanol",
                    "hazard_class": "Toxic",
                    "exposure_limits": "200 ppm (8-hour TWA)",
                    "health_effects": ["Blindness", "Central nervous system depression"]
                }
            ]
            
            for compound in safety_data:
                self.knowledge_base[f"safety_{compound['compound']}"] = {
                    "content": f"Compound: {compound['compound']}\nHazard Class: {compound['hazard_class']}\nHealth Effects: {', '.join(compound['health_effects'])}",
                    "metadata": compound,
                    "type": "safety_data"
                }
                
        except Exception as e:
            logger.error(f"Error loading chemical safety data: {e}")
    
    def load_clinical_trial_data(self):
        """Load clinical trial toxicity data"""
        try:
            # Load from clinical trial databases
            trial_data = [
                {
                    "trial_id": "NCT12345678",
                    "compound": "Experimental Drug X",
                    "adverse_events": ["Nausea", "Headache"],
                    "severity": "Mild to moderate",
                    "frequency": "Common"
                }
            ]
            
            for trial in trial_data:
                self.knowledge_base[f"trial_{trial['trial_id']}"] = {
                    "content": f"Trial: {trial['trial_id']}\nCompound: {trial['compound']}\nAdverse Events: {', '.join(trial['adverse_events'])}",
                    "metadata": trial,
                    "type": "clinical_trial"
                }
                
        except Exception as e:
            logger.error(f"Error loading clinical trial data: {e}")
    
    def load_toxcast_database(self):
        """Load ToxCast database for high-throughput toxicity screening"""
        try:
            # Load ToxCast data (simulated)
            toxcast_data = {
                "assays": ["Nuclear receptor", "Stress response", "Cell cycle"],
                "compounds": ["BPA", "Phthalates", "Pesticides"],
                "endpoints": ["Estrogen receptor", "Androgen receptor", "Thyroid receptor"]
            }
            
            self.research_databases["toxcast"] = toxcast_data
            
        except Exception as e:
            logger.error(f"Error loading ToxCast database: {e}")
    
    def load_chembl_database(self):
        """Load ChEMBL database for bioactivity data"""
        try:
            # Load ChEMBL data (simulated)
            chembl_data = {
                "targets": ["Cytochrome P450", "hERG channel", "Liver enzymes"],
                "activities": ["Inhibition", "Activation", "Modulation"],
                "compounds": ["Drug-like molecules", "Natural products", "Synthetic compounds"]
            }
            
            self.research_databases["chembl"] = chembl_data
            
        except Exception as e:
            logger.error(f"Error loading ChEMBL database: {e}")
    
    def load_pubchem_toxicity(self):
        """Load PubChem toxicity data"""
        try:
            # Load PubChem toxicity data (simulated)
            pubchem_data = {
                "toxicity_endpoints": ["LD50", "LC50", "EC50"],
                "species": ["Mouse", "Rat", "Human"],
                "routes": ["Oral", "Intravenous", "Inhalation"]
            }
            
            self.research_databases["pubchem"] = pubchem_data
            
        except Exception as e:
            logger.error(f"Error loading PubChem toxicity data: {e}")
    
    def load_fda_safety_data(self):
        """Load FDA drug safety data"""
        try:
            # Load FDA safety data (simulated)
            fda_data = {
                "adverse_events": ["Liver injury", "Cardiotoxicity", "Neurotoxicity"],
                "black_box_warnings": ["Severe liver damage", "Heart failure", "Suicidal thoughts"],
                "post_market_surveillance": ["FAERS database", "MedWatch reports"]
            }
            
            self.research_databases["fda"] = fda_data
            
        except Exception as e:
            logger.error(f"Error loading FDA safety data: {e}")
    
    def retrieve_toxicity_info(self, query: str, compound_smiles: str = None, 
                              top_k: int = 5, use_ensemble: bool = True) -> List[Dict]:
        """Enhanced retrieval with compound-specific context and multiple strategies"""
        try:
            # Check cache first
            cache_key = f"{query}_{compound_smiles}_{top_k}_{use_ensemble}"
            if cache_key in self.retrieval_cache:
                return self.retrieval_cache[cache_key]
            
            retrieved_info = []
            
            # Add compound-specific context to query
            if compound_smiles:
                compound_context = self.get_compound_context(compound_smiles)
                enhanced_query = f"{query} {compound_context}"
            else:
                enhanced_query = query
            
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
                    "relevance_score": self.calculate_toxicity_relevance_score(enhanced_query, doc.page_content, compound_smiles),
                    "evidence_level": doc.metadata.get("evidence_level", "unknown")
                }
                retrieved_info.append(info)
            
            # Sort by relevance score
            retrieved_info.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Cache results
            self.retrieval_cache[cache_key] = retrieved_info[:top_k]
            
            logger.info(f"Retrieved {len(retrieved_info)} relevant documents for toxicity query: {query[:50]}...")
            return retrieved_info[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving toxicity information: {e}")
            return []
    
    def get_compound_context(self, smiles: str) -> str:
        """Get chemical context for compound to enhance retrieval"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            
            # Extract chemical properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            
            # Get functional groups
            functional_groups = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N':
                    functional_groups.append("amine")
                elif atom.GetSymbol() == 'O':
                    functional_groups.append("alcohol/ether")
                elif atom.GetSymbol() == 'S':
                    functional_groups.append("thiol/sulfide")
            
            context = f"MW:{mw:.1f} LogP:{logp:.2f} HBD:{hbd} HBA:{hba} TPSA:{tpsa:.1f} Groups:{' '.join(set(functional_groups))}"
            return context
            
        except Exception as e:
            logger.error(f"Error getting compound context: {e}")
            return ""
    
    def calculate_toxicity_relevance_score(self, query: str, content: str, compound_smiles: str = None) -> float:
        """Calculate relevance score for toxicity-related content"""
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
            
            # Boost score for toxicity-related terms
            toxicity_terms = [
                "toxicity", "toxic", "poison", "hazard", "risk", "safety",
                "adverse", "side effect", "injury", "damage", "harm",
                "carcinogen", "mutagen", "teratogen", "hepatotoxic", "nephrotoxic"
            ]
            toxicity_boost = sum(1 for term in toxicity_terms if term in content.lower())
            relevance += toxicity_boost * 0.15
            
            # Boost for compound-specific information
            if compound_smiles and compound_smiles in content:
                relevance += 0.2
            
            # Boost for scientific/medical sources
            scientific_indicators = ["study", "research", "clinical", "trial", "experiment", "analysis"]
            scientific_boost = sum(1 for indicator in scientific_indicators if indicator in content.lower())
            relevance += scientific_boost * 0.05
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating toxicity relevance score: {e}")
            return 0.0
    
    def get_research_evidence(self, compound_smiles: str, toxicity_class: str) -> Dict[str, Any]:
        """Get research evidence for compound toxicity"""
        try:
            evidence = {
                "literature_references": [],
                "clinical_data": [],
                "mechanistic_studies": [],
                "regulatory_status": [],
                "similar_compounds": []
            }
            
            # Search for literature references
            lit_query = f"toxicity {toxicity_class} {compound_smiles}"
            literature = self.retrieve_toxicity_info(lit_query, compound_smiles, top_k=3)
            evidence["literature_references"] = literature
            
            # Search for clinical data
            clinical_query = f"clinical trial safety {compound_smiles}"
            clinical = self.retrieve_toxicity_info(clinical_query, compound_smiles, top_k=2)
            evidence["clinical_data"] = clinical
            
            # Search for mechanistic studies
            mech_query = f"mechanism toxicity {compound_smiles}"
            mechanistic = self.retrieve_toxicity_info(mech_query, compound_smiles, top_k=2)
            evidence["mechanistic_studies"] = mechanistic
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error getting research evidence: {e}")
            return {}

class MolecularDescriptorCalculator:
    """Calculate molecular descriptors for drug compounds"""
    
    def __init__(self):
        self.descriptor_functions = {
            'MolWt': Descriptors.MolWt,
            'LogP': Descriptors.MolLogP,
            'NumHDonors': Descriptors.NumHDonors,
            'NumHAcceptors': Descriptors.NumHAcceptors,
            'TPSA': Descriptors.TPSA,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'FractionCsp3': Descriptors.FractionCsp3,
            'HeavyAtomCount': Descriptors.HeavyAtomCount,
            'RingCount': Descriptors.RingCount
        }
    
    def calculate_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular descriptors for a given SMILES string"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            
            descriptors = {}
            for name, func in self.descriptor_functions.items():
                try:
                    descriptors[name] = float(func(mol))
                except:
                    descriptors[name] = 0.0
            
            # Calculate additional descriptors
            descriptors.update(self.calculate_fingerprints(mol))
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating descriptors for {smiles}: {e}")
            return {}
    
    def calculate_fingerprints(self, mol) -> Dict[str, float]:
        """Calculate molecular fingerprints"""
        fingerprints = {}
        
        try:
            # MACCS keys
            maccs = MACCSkeys.GenMACCSKeys(mol)
            fingerprints['MACCS_Keys'] = maccs.ToBitString()
            
            # Morgan fingerprints
            morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints['Morgan_FP'] = morgan.ToBitString()
            
            # Atom pairs
            pairs = Pairs.GetAtomPairFingerprint(mol)
            fingerprints['Atom_Pairs'] = pairs.ToBitString()
            
        except Exception as e:
            print(f"Error calculating fingerprints: {e}")
        
        return fingerprints

class ToxicityDataset(Dataset):
    """Custom dataset for toxicity classification"""
    
    def __init__(self, compounds: List[DrugCompound], labels: List[str], transform=None):
        self.compounds = compounds
        self.labels = labels
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
    
    def __len__(self):
        return len(self.compounds)
    
    def __getitem__(self, idx):
        compound = self.compounds[idx]
        label = self.labels[idx]
        
        # Convert descriptors to tensor
        descriptor_values = list(compound.descriptors.values())
        features = torch.tensor(descriptor_values, dtype=torch.float32)
        
        # Convert label to tensor
        label_idx = self.label_encoder.transform([label])[0]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label_tensor

class ToxicityCNN(nn.Module):
    """CNN model for molecular structure analysis"""
    
    def __init__(self, input_size: int, num_classes: int):
        super(ToxicityCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Reshape input for 1D convolution
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for molecular graph analysis"""
    
    def __init__(self, num_node_features: int, num_classes: int):
        super(GraphNeuralNetwork, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return x

class DrugToxicityClassifier:
    """Main drug toxicity classification system"""
    
    def __init__(self):
        self.rag_system = ToxicityRAGSystem()
        self.descriptor_calculator = MolecularDescriptorCalculator()
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize different classification models"""
        
        # Traditional ML models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        self.models['svm'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Deep learning models
        self.models['cnn'] = ToxicityCNN(input_size=2048, num_classes=4)
        self.models['gnn'] = GraphNeuralNetwork(num_node_features=74, num_classes=4)
    
    def prepare_molecular_data(self, smiles: str) -> Tuple[torch.Tensor, GraphData]:
        """Prepare molecular data for different model types"""
        
        # Calculate descriptors
        descriptors = self.descriptor_calculator.calculate_descriptors(smiles)
        descriptor_tensor = torch.tensor(list(descriptors.values()), dtype=torch.float32)
        
        # Create molecular graph
        mol = Chem.MolFromSmiles(smiles)
        graph_data = self.create_molecular_graph(mol)
        
        return descriptor_tensor, graph_data
    
    def create_molecular_graph(self, mol) -> GraphData:
        """Create molecular graph representation"""
        
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = self.get_atom_features(atom)
            atom_features.append(features)
        
        # Get edge features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([start, end])
            edge_index.append([end, start])  # Undirected graph
            
            bond_features = self.get_bond_features(bond)
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)
        
        x = torch.tensor(atom_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def get_atom_features(self, atom) -> List[float]:
        """Get atom features for graph representation"""
        features = []
        
        # Atom type (one-hot encoding)
        atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        atom_type = atom.GetSymbol()
        for atom_t in atom_types:
            features.append(1.0 if atom_t == atom_type else 0.0)
        
        # Additional features
        features.extend([
            atom.GetDegree(),
            atom.GetImplicitValence(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
            atom.GetMass() / 100.0,  # Normalized mass
        ])
        
        return features
    
    def get_bond_features(self, bond) -> List[float]:
        """Get bond features for graph representation"""
        bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, 
                     Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
        
        features = []
        for bond_type in bond_types:
            features.append(1.0 if bond.GetBondType() == bond_type else 0.0)
        
        features.append(bond.GetIsConjugated())
        features.append(bond.IsInRing())
        
        return features
    
    def train_models(self, training_data: List[DrugCompound], labels: List[str]):
        """Train all classification models"""
        
        # Prepare features
        X = []
        for compound in training_data:
            descriptor_values = list(compound.descriptors.values())
            X.append(descriptor_values)
        
        X = np.array(X)
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train traditional ML models
        for name, model in self.models.items():
            if name in ['random_forest', 'gradient_boosting', 'svm']:
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = model.score(X_test_scaled, y_test)
                print(f"{name} accuracy: {accuracy:.4f}")
    
    def predict_toxicity(self, smiles: str, compound_name: str = "Unknown") -> ToxicityPrediction:
        """Predict toxicity for a given compound"""
        
        try:
            # Calculate molecular descriptors
            descriptors = self.descriptor_calculator.calculate_descriptors(smiles)
            
            # Prepare features for prediction
            feature_values = list(descriptors.values())
            features_scaled = self.scaler.transform([feature_values])
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                if name in ['random_forest', 'gradient_boosting', 'svm']:
                    pred = model.predict(features_scaled)[0]
                    prob = model.predict_proba(features_scaled)[0]
                    
                    predictions[name] = pred
                    probabilities[name] = prob
            
            # Ensemble prediction
            ensemble_pred = self.ensemble_predict(predictions, probabilities)
            
            # Retrieve relevant toxicity information
            toxicity_info = self.rag_system.retrieve_toxicity_info(
                f"toxicity {compound_name} {smiles}", top_k=3
            )
            
            # Generate risk factors and recommendations
            risk_factors = self.identify_risk_factors(descriptors, toxicity_info)
            recommendations = self.generate_recommendations(ensemble_pred, risk_factors)
            
            return ToxicityPrediction(
                compound_name=compound_name,
                smiles=smiles,
                predicted_toxicity_class=ensemble_pred['class'],
                confidence_score=ensemble_pred['confidence'],
                toxicity_probabilities=ensemble_pred['probabilities'],
                risk_factors=risk_factors,
                recommendations=recommendations,
                sources=[info['source'] for info in toxicity_info]
            )
            
        except Exception as e:
            print(f"Error predicting toxicity: {e}")
            raise
    
    def ensemble_predict(self, predictions: Dict, probabilities: Dict) -> Dict:
        """Combine predictions from multiple models"""
        
        # Weighted voting based on model performance
        model_weights = {
            'random_forest': 0.4,
            'gradient_boosting': 0.3,
            'svm': 0.3
        }
        
        # Calculate weighted probabilities
        weighted_probs = np.zeros(len(self.label_encoder.classes_))
        for name, prob in probabilities.items():
            if name in model_weights:
                weighted_probs += prob * model_weights[name]
        
        # Get final prediction
        final_class_idx = np.argmax(weighted_probs)
        final_class = self.label_encoder.inverse_transform([final_class_idx])[0]
        confidence = np.max(weighted_probs)
        
        return {
            'class': final_class,
            'confidence': confidence,
            'probabilities': {
                self.label_encoder.inverse_transform([i])[0]: prob 
                for i, prob in enumerate(weighted_probs)
            }
        }
    
    def identify_risk_factors(self, descriptors: Dict[str, float], 
                            toxicity_info: List[Dict]) -> List[str]:
        """Identify potential risk factors based on molecular properties"""
        
        risk_factors = []
        
        # Check molecular weight
        if descriptors.get('MolWt', 0) > 500:
            risk_factors.append("High molecular weight (>500 Da)")
        
        # Check LogP
        if descriptors.get('LogP', 0) > 5:
            risk_factors.append("High lipophilicity (LogP > 5)")
        
        # Check hydrogen bond donors
        if descriptors.get('NumHDonors', 0) > 5:
            risk_factors.append("High number of hydrogen bond donors (>5)")
        
        # Check aromatic rings
        if descriptors.get('NumAromaticRings', 0) > 3:
            risk_factors.append("High number of aromatic rings (>3)")
        
        # Add risk factors from literature
        for info in toxicity_info:
            if "hepatotoxicity" in info['content'].lower():
                risk_factors.append("Potential hepatotoxicity based on literature")
            if "cardiotoxicity" in info['content'].lower():
                risk_factors.append("Potential cardiotoxicity based on literature")
        
        return risk_factors
    
    def generate_recommendations(self, prediction: Dict, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on toxicity prediction and risk factors"""
        
        recommendations = []
        
        # General recommendations
        recommendations.append("Conduct comprehensive toxicity testing")
        recommendations.append("Monitor for adverse effects in clinical trials")
        
        # Class-specific recommendations
        toxicity_class = prediction['class']
        if toxicity_class == 'high':
            recommendations.extend([
                "Implement strict safety monitoring protocols",
                "Consider alternative compound development",
                "Conduct extensive preclinical safety studies"
            ])
        elif toxicity_class == 'moderate':
            recommendations.extend([
                "Implement standard safety monitoring",
                "Conduct thorough preclinical evaluation",
                "Monitor specific organ systems based on risk factors"
            ])
        elif toxicity_class == 'low':
            recommendations.extend([
                "Standard safety protocols sufficient",
                "Monitor for unexpected adverse effects",
                "Continue with normal development process"
            ])
        
        # Risk factor specific recommendations
        for risk_factor in risk_factors:
            if "hepatotoxicity" in risk_factor:
                recommendations.append("Monitor liver function tests closely")
            if "cardiotoxicity" in risk_factor:
                recommendations.append("Monitor cardiac function and ECG")
            if "high molecular weight" in risk_factor:
                recommendations.append("Consider bioavailability and absorption")
        
        return recommendations

# Pydantic models for API
class ToxicityPredictionRequest(BaseModel):
    smiles: str
    compound_name: str = "Unknown"
    include_descriptors: bool = False

class ToxicityPredictionResponse(BaseModel):
    compound_name: str
    smiles: str
    predicted_toxicity_class: str
    confidence_score: float
    toxicity_probabilities: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    sources: List[str]
    molecular_descriptors: Optional[Dict[str, float]] = None

# Initialize classifier
toxicity_classifier = DrugToxicityClassifier()

@app.post("/predict", response_model=ToxicityPredictionResponse)
async def predict_toxicity(request: ToxicityPredictionRequest):
    """Predict toxicity for a given compound"""
    try:
        prediction = toxicity_classifier.predict_toxicity(
            request.smiles, request.compound_name
        )
        
        response_data = {
            "compound_name": prediction.compound_name,
            "smiles": prediction.smiles,
            "predicted_toxicity_class": prediction.predicted_toxicity_class,
            "confidence_score": prediction.confidence_score,
            "toxicity_probabilities": prediction.toxicity_probabilities,
            "risk_factors": prediction.risk_factors,
            "recommendations": prediction.recommendations,
            "sources": prediction.sources
        }
        
        if request.include_descriptors:
            descriptors = toxicity_classifier.descriptor_calculator.calculate_descriptors(
                request.smiles
            )
            response_data["molecular_descriptors"] = descriptors
        
        return ToxicityPredictionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict_toxicity(file: UploadFile = File(...)):
    """Batch predict toxicity for multiple compounds"""
    try:
        # Read CSV file
        df = pd.read_csv(file.file)
        
        results = []
        for _, row in df.iterrows():
            try:
                prediction = toxicity_classifier.predict_toxicity(
                    row['smiles'], row.get('name', 'Unknown')
                )
                results.append({
                    "compound_name": prediction.compound_name,
                    "smiles": prediction.smiles,
                    "predicted_toxicity_class": prediction.predicted_toxicity_class,
                    "confidence_score": prediction.confidence_score,
                    "risk_factors": prediction.risk_factors
                })
            except Exception as e:
                results.append({
                    "compound_name": row.get('name', 'Unknown'),
                    "smiles": row['smiles'],
                    "error": str(e)
                })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 