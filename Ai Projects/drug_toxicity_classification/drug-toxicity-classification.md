# Drug Toxicity Classification using ML & CNNs (Classification)

## Project Overview
A comprehensive drug toxicity classification system that leverages advanced machine learning algorithms, deep learning architectures, and Retrieval-Augmented Generation (RAG) to predict the toxicity levels of chemical compounds. This system combines molecular descriptor calculations, graph neural networks, and convolutional neural networks with extensive research literature integration to provide accurate toxicity predictions with detailed mechanistic insights and regulatory compliance information.

The system employs a multi-modal approach that processes molecular structures through both traditional cheminformatics methods and modern deep learning techniques, while simultaneously accessing the latest toxicity research literature, clinical trial data, and regulatory guidelines through its RAG pipeline. This integration ensures that predictions are not only based on structural similarity but also informed by the most current scientific understanding of toxicity mechanisms and clinical evidence.

## RAG Architecture Overview

### Enhanced Research Literature Integration
The system implements a sophisticated RAG pipeline that integrates multiple specialized research databases including PubMed toxicity articles, ToxCast high-throughput screening data, ChEMBL bioactivity databases, PubChem toxicity endpoints, and FDA drug safety information. The RAG system employs ensemble retrieval strategies combining vector similarity search, keyword-based retrieval (BM25), and semantic matching to ensure comprehensive coverage of toxicity-related literature and research findings.

The knowledge base is structured hierarchically with specialized collections for different toxicity endpoints including hepatotoxicity, cardiotoxicity, genotoxicity, and organ-specific toxicities. Each research source is tagged with metadata including publication date, evidence level, study design, and clinical relevance, enabling intelligent source ranking and evidence-based toxicity assessment. The system also maintains real-time updates from regulatory agencies and clinical trial databases to ensure predictions reflect the most current safety information.

### Molecular Context-Aware Retrieval
The RAG system incorporates molecular context awareness by extracting chemical properties, functional groups, and structural features from compound SMILES strings to enhance retrieval relevance. This molecular context is used to query research literature for similar compounds, known toxicity mechanisms, and relevant clinical data, providing a comprehensive understanding of the compound's potential toxicity profile.

The system employs advanced chemical entity recognition to identify molecular substructures, pharmacophores, and toxicophores in research literature, enabling precise matching between query compounds and relevant toxicity studies. This molecular-aware retrieval ensures that toxicity predictions are supported by the most relevant scientific evidence and mechanistic understanding.

## Key Features
- **Advanced RAG Integration**: Multi-database research literature access with molecular context awareness
- **Multi-Modal ML Models**: CNN, GNN, and ensemble methods for comprehensive toxicity prediction
- **Molecular Descriptor Analysis**: Comprehensive cheminformatics calculations and fingerprinting
- **Research Evidence Integration**: Real-time access to PubMed, ToxCast, ChEMBL, and FDA databases
- **Regulatory Compliance**: Integration with ICH guidelines and FDA safety requirements
- **Mechanistic Insights**: Detailed toxicity mechanism analysis and risk factor identification
- **Clinical Trial Data**: Access to clinical safety data and adverse event information
- **Ensemble Prediction**: Multiple model consensus for improved accuracy and reliability
- **Source Attribution**: Transparent citation of research sources and evidence levels

## Technology Stack
- **Deep Learning**: PyTorch with CNN and Graph Neural Network architectures
- **Cheminformatics**: RDKit for molecular descriptor calculation and fingerprinting
- **RAG Framework**: LangChain with ensemble retrieval and molecular context integration
- **Vector Database**: ChromaDB for research literature embeddings storage
- **Research Databases**: PubMed API, ToxCast, ChEMBL, PubChem, FDA safety data
- **ML Algorithms**: Random Forest, Gradient Boosting, SVM for ensemble prediction
- **Graph Neural Networks**: PyTorch Geometric for molecular graph processing
- **FastAPI**: RESTful API for system integration and batch processing
- **Chemical Visualization**: Molecular structure rendering and toxicity pathway mapping

## Complete System Flow

### Phase 1: Molecular Data Processing and Research Context Extraction
The system begins by receiving chemical compound information in SMILES format and processes it through a comprehensive molecular analysis pipeline. The molecular descriptor calculator extracts over 200 chemical properties including molecular weight, logP, hydrogen bond donors/acceptors, topological polar surface area, and various molecular fingerprints. Simultaneously, the RAG system extracts molecular context including functional groups, structural features, and chemical properties to enhance research literature retrieval.

The system then queries multiple research databases including PubMed for toxicity literature, ToxCast for high-throughput screening data, ChEMBL for bioactivity information, and FDA databases for clinical safety data. The molecular context is used to identify similar compounds, known toxicity mechanisms, and relevant clinical studies, providing a comprehensive research foundation for toxicity prediction. The retrieved information is processed through relevance scoring that considers molecular similarity, study quality, and clinical relevance.

### Phase 2: Multi-Model Toxicity Prediction with Research Evidence Integration
Once the molecular data and research context are prepared, the system employs multiple machine learning models including Random Forest, Gradient Boosting, Support Vector Machines, Convolutional Neural Networks, and Graph Neural Networks. Each model processes the molecular descriptors and structural information through different approaches, with CNNs handling molecular fingerprint vectors and GNNs processing molecular graphs with atom and bond features.

The RAG system simultaneously provides research evidence including literature references, clinical trial data, mechanistic studies, and regulatory information to inform the prediction process. The ensemble prediction combines results from all models with research evidence weighting to generate a comprehensive toxicity assessment. The system also identifies specific risk factors based on molecular features and research literature, providing detailed mechanistic insights into potential toxicity pathways.

### Phase 3: Comprehensive Toxicity Analysis and Regulatory Compliance Assessment
The final phase generates a comprehensive toxicity report that includes predicted toxicity class, confidence scores, detailed probability distributions, identified risk factors, and specific recommendations. The system integrates regulatory compliance information by checking against ICH guidelines, FDA requirements, and international safety standards to ensure predictions meet regulatory requirements.

The research evidence is synthesized to provide detailed mechanistic insights, clinical relevance assessment, and safety recommendations. The system generates comprehensive reports that include source attribution, evidence levels, and regulatory status, enabling informed decision-making in drug development and safety assessment. Continuous learning mechanisms update the models and knowledge base with new research findings and clinical data.

## RAG Implementation Details

### Research Database Integration
- **PubMed API**: Real-time access to toxicity research articles and clinical studies
- **ToxCast Database**: High-throughput screening data for toxicity endpoints
- **ChEMBL Database**: Bioactivity data and target interaction information
- **PubChem Toxicity**: Comprehensive toxicity endpoint data and safety information
- **FDA Safety Data**: Clinical trial safety data and post-market surveillance
- **Regulatory Guidelines**: ICH, FDA, and international safety requirements

### Molecular-Aware Retrieval
- **Chemical Entity Recognition**: Identification of molecular substructures and pharmacophores
- **Similarity-Based Retrieval**: Finding similar compounds and their toxicity profiles
- **Mechanistic Literature**: Access to toxicity mechanism studies and pathway analysis
- **Clinical Evidence**: Integration of clinical trial safety data and adverse events
- **Regulatory Context**: Compliance with safety guidelines and regulatory requirements

### Evidence Synthesis
- **Multi-Source Integration**: Combining literature, clinical, and regulatory evidence
- **Evidence Level Assessment**: Quality scoring of research sources and study design
- **Mechanistic Insights**: Detailed analysis of toxicity pathways and molecular mechanisms
- **Risk Factor Identification**: Automated detection of structural and functional risk factors
- **Regulatory Compliance**: Assessment against international safety standards

## Use Cases
- Drug development toxicity screening with research evidence integration
- Chemical safety assessment with regulatory compliance checking
- Pharmaceutical compound evaluation with mechanistic insights
- Environmental chemical toxicity prediction with literature support
- Drug repurposing safety analysis with clinical data integration
- Regulatory submission support with evidence-based documentation
- Academic research with comprehensive literature access
- Industrial chemical safety with multi-database validation

## Implementation Areas
- Advanced molecular descriptor calculation with RDKit integration
- Multi-modal deep learning model development with PyTorch
- Comprehensive RAG pipeline with research database integration
- Ensemble prediction algorithms with evidence weighting
- Regulatory compliance assessment with guideline integration
- Chemical visualization and toxicity pathway mapping
- Real-time research database updates and model retraining
- Clinical trial data integration and safety monitoring

## Expected Outcomes
- Highly accurate toxicity predictions with research evidence support
- Comprehensive mechanistic insights with literature integration
- Regulatory compliance assessment with guideline adherence
- Real-time access to latest toxicity research and clinical data
- Detailed risk factor identification with molecular basis
- Evidence-based safety recommendations with source attribution
- Scalable drug development support with batch processing
- Continuous learning with research literature updates 