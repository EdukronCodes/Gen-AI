# Module 11: Frameworks, Libraries & Platforms Overview

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Class:** 19

---

## Class 19: Tools, Frameworks & Platforms

### Topics Covered

- **Frameworks:** PyTorch, TensorFlow, Hugging Face Transformers
- **Libraries:** LangChain, LlamaIndex, Gradio, Streamlit, MLflow
- **Platforms:** Hugging Face Hub, Azure OpenAI, Vertex AI, Ollama
- **Databases:** Pinecone, ChromaDB, Weaviate, Milvus
- **Utilities:** OpenAI API, Anthropic API, Replicate, Modal

### Learning Objectives

By the end of this class, students will be able to:
- Understand the GenAI ecosystem
- Choose appropriate tools for different tasks
- Navigate different platforms and services
- Integrate multiple tools in workflows
- Make informed tool selection decisions

### Core Concepts

#### Deep Learning Frameworks

**PyTorch:**
- **Type:** Deep learning framework
- **Developer:** Meta (Facebook)
- **Key Features:**
  - Dynamic computation graphs
  - Pythonic API
  - Strong research community
  - Excellent for prototyping
- **Use Cases:**
  - Model development
  - Research
  - Fine-tuning
  - Custom architectures
- **Ecosystem:**
  - TorchServe (deployment)
  - PyTorch Lightning (training)
  - Hugging Face (integrated)

**TensorFlow:**
- **Type:** Deep learning framework
- **Developer:** Google
- **Key Features:**
  - Static computation graphs
  - Production-ready
  - TensorBoard (visualization)
  - Strong deployment tools
- **Use Cases:**
  - Production deployment
  - Large-scale training
  - TensorFlow.js (web)
- **Ecosystem:**
  - Keras (high-level API)
  - TensorFlow Serving
  - TensorFlow Lite (mobile)

**Hugging Face Transformers:**
- **Type:** NLP library and platform
- **Developer:** Hugging Face
- **Key Features:**
  - Pre-trained models
  - Easy model loading
  - Tokenizers
  - Model hub
- **Use Cases:**
  - Quick model access
  - Fine-tuning
  - Model sharing
- **Models:**
  - Thousands of pre-trained models
  - All major architectures
  - Community contributions

#### Application Frameworks

**LangChain:**
- **Type:** Application framework
- **Focus:** LLM application orchestration
- **Key Features:**
  - Chains
  - Agents
  - Tools
  - Memory
- **Use Cases:**
  - Complex workflows
  - Agentic systems
  - RAG applications
- **Integrations:**
  - All major LLMs
  - Vector databases
  - Tools and APIs

**LlamaIndex:**
- **Type:** Data framework
- **Focus:** Data ingestion and indexing
- **Key Features:**
  - Index types
  - Query interface
  - Data connectors
- **Use Cases:**
  - Knowledge bases
  - Document Q&A
  - RAG systems
- **Strengths:**
  - Data-centric
  - Flexible indexing

**Gradio:**
- **Type:** UI framework
- **Focus:** Quick ML interfaces
- **Key Features:**
  - Simple Python API
  - Automatic UI generation
  - Sharing capabilities
- **Use Cases:**
  - Demos
  - Prototyping
  - User testing

**Streamlit:**
- **Type:** Web app framework
- **Focus:** Data applications
- **Key Features:**
  - Python-based
  - Interactive widgets
  - Easy deployment
- **Use Cases:**
  - Data apps
  - Dashboards
  - Interactive demos

**MLflow:**
- **Type:** ML lifecycle management
- **Focus:** Experiment tracking and model management
- **Key Features:**
  - Experiment tracking
  - Model registry
  - Model versioning
- **Use Cases:**
  - Experiment management
  - Model deployment
  - Reproducibility

#### Cloud Platforms

**Hugging Face Hub:**
- **Type:** Model hosting and sharing
- **Features:**
  - Model repository
  - Dataset hosting
  - Spaces (app hosting)
  - Inference API
- **Use Cases:**
  - Model sharing
  - Model deployment
  - Collaboration
- **Pricing:** Free tier available

**Azure OpenAI:**
- **Type:** Managed OpenAI service
- **Provider:** Microsoft Azure
- **Features:**
  - GPT models (GPT-3.5, GPT-4)
  - Embeddings
  - Enterprise security
  - Regional availability
- **Use Cases:**
  - Enterprise deployments
  - Azure integration
  - Compliance requirements

**Google Vertex AI:**
- **Type:** ML platform
- **Provider:** Google Cloud
- **Features:**
  - Gemini models
  - PaLM models
  - Custom model training
  - MLOps tools
- **Use Cases:**
  - GCP deployments
  - Custom models
  - Enterprise ML

**AWS Bedrock:**
- **Type:** Managed foundation models
- **Provider:** Amazon Web Services
- **Features:**
  - Multiple model providers
  - Claude, Llama, Titan
  - Serverless inference
  - AWS integration
- **Use Cases:**
  - AWS deployments
  - Multi-model access
  - Enterprise AWS customers

**Ollama:**
- **Type:** Local LLM runner
- **Features:**
  - Run models locally
  - Multiple models
  - Simple API
  - Privacy-focused
- **Use Cases:**
  - Local development
  - Privacy-sensitive apps
  - Cost-effective inference

#### Vector Databases

**Pinecone:**
- **Type:** Managed vector database
- **Features:**
  - Serverless
  - Auto-scaling
  - Global distribution
  - Simple API
- **Use Cases:**
  - Production RAG
  - Large-scale search
  - Managed service needs

**ChromaDB:**
- **Type:** Open-source vector database
- **Features:**
  - Lightweight
  - Easy to use
  - Python-first
  - Local or hosted
- **Use Cases:**
  - Development
  - Small-medium scale
  - Cost-sensitive projects

**Weaviate:**
- **Type:** Open-source vector database
- **Features:**
  - GraphQL API
  - Built-in vectorization
  - Multi-tenancy
  - Advanced queries
- **Use Cases:**
  - Complex search
  - Graph features
  - Enterprise search

**Milvus:**
- **Type:** Open-source vector database
- **Features:**
  - High performance
  - Scalable
  - Multiple index types
  - Enterprise features
- **Use Cases:**
  - Large-scale deployments
  - Enterprise applications
  - High-performance needs

#### API Services

**OpenAI API:**
- **Provider:** OpenAI
- **Models:**
  - GPT-3.5, GPT-4
  - Embeddings
  - Whisper (speech)
  - DALL-E (images)
- **Features:**
  - Simple API
  - Wide adoption
  - Good documentation

**Anthropic API:**
- **Provider:** Anthropic
- **Models:**
  - Claude 3 (Haiku, Sonnet, Opus)
- **Features:**
  - Long context windows
  - Safety-focused
  - Good for long documents

**Replicate:**
- **Type:** Model hosting platform
- **Features:**
  - Run any model
  - Pay-per-use
  - Simple API
  - Community models
- **Use Cases:**
  - Model access
  - Quick prototyping
  - No infrastructure

**Modal:**
- **Type:** Serverless compute platform
- **Features:**
  - Run code on cloud
  - Pay-per-use
  - GPU support
  - Simple deployment
- **Use Cases:**
  - Serverless functions
  - GPU workloads
  - Cost-effective compute

#### Tool Selection Guide

**For Model Development:**
- PyTorch or TensorFlow
- Hugging Face Transformers
- Jupyter Notebooks

**For Application Building:**
- LangChain or LlamaIndex
- Gradio or Streamlit
- FastAPI or Flask

**For Model Hosting:**
- Hugging Face Hub (open-source)
- Azure OpenAI (enterprise, Azure)
- Vertex AI (GCP, custom models)
- AWS Bedrock (AWS, multi-provider)

**For Vector Search:**
- ChromaDB (development, small scale)
- Pinecone (production, managed)
- Weaviate (complex queries)
- Milvus (large scale, enterprise)

**For Experiment Management:**
- MLflow
- Weights & Biases
- TensorBoard

**For Local Development:**
- Ollama (local models)
- ChromaDB (local vector DB)
- Docker (containerization)

#### Integration Patterns

**Common Workflows:**

**1. RAG Pipeline:**
```
Documents → LangChain → ChromaDB → OpenAI API → Response
```

**2. Fine-tuning:**
```
Data → Hugging Face → PyTorch → Fine-tune → Hugging Face Hub
```

**3. Production Deployment:**
```
Model → FastAPI → Docker → Cloud Platform → Monitoring (MLflow)
```

**4. Local Development:**
```
Ollama → LangChain → ChromaDB → Gradio → Local UI
```

### Readings

- Documentation for major frameworks and platforms:
  - [PyTorch Documentation](https://pytorch.org/docs/)
  - [Hugging Face Documentation](https://huggingface.co/docs)
  - [LangChain Documentation](https://python.langchain.com/)
  - [Platform comparison guides](https://www.pinecone.io/learn/)

 

### Additional Resources

- [Hugging Face Hub](https://huggingface.co/)
- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Ollama](https://ollama.ai/)
- [Replicate](https://replicate.com/)
- [Modal](https://modal.com/)

### Practical Code Examples

#### Tool Integration Example

```python
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun

class IntegratedRAGSystem:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.vectorstore = Chroma(persist_directory="./chroma_db")
        self.search = DuckDuckGoSearchRun()
        
        # Create tools
        self.tools = [
            Tool(
                name="Document Search",
                func=self._search_documents,
                description="Search internal documents for information"
            ),
            Tool(
                name="Web Search",
                func=self.search.run,
                description="Search the web for current information"
            )
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )
    
    def _search_documents(self, query: str) -> str:
        """Search internal documents"""
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])
    
    def query(self, question: str) -> str:
        """Query the integrated system"""
        return self.agent.run(question)

# Usage
system = IntegratedRAGSystem()
answer = system.query("What is the latest information about AI?")
print(answer)
```

**Pro Tip:** Combine multiple tools to create more powerful systems. Use agents to orchestrate tool usage based on query type.

**Common Pitfall:** Using too many tools without proper orchestration can lead to confusion and poor results. Start with essential tools and add more as needed.

### Quick Reference Guide

#### Tool Selection Matrix

| Use Case | Recommended Tools | Why |
|----------|------------------|-----|
| Development | LangChain, ChromaDB, Gradio | Fast iteration, easy setup |
| Production | FastAPI, Pinecone, Kubernetes | Scalable, reliable |
| Research | PyTorch, Hugging Face, Jupyter | Flexibility, experimentation |
| Local/Privacy | Ollama, ChromaDB, Local models | Data privacy, cost control |

#### Platform Comparison

| Platform | Best For | Cost | Ease of Use |
|----------|----------|------|-------------|
| Hugging Face Hub | Open-source models | Free/Paid | Easy |
| Azure OpenAI | Enterprise Azure customers | Pay-per-use | Easy |
| Vertex AI | GCP customers | Pay-per-use | Moderate |
| AWS Bedrock | AWS customers | Pay-per-use | Moderate |

### Key Takeaways

1. Rich ecosystem of tools for different needs
2. Framework choice depends on use case and requirements
3. Cloud platforms provide managed services for production
4. Vector databases enable semantic search at scale
5. API services provide easy access to powerful models
6. Local tools (Ollama) enable privacy and cost control
7. Integration of multiple tools creates powerful systems
8. Tool selection should consider: cost, ease of use, scale, features
9. Start with essential tools and expand as needed
10. Proper integration and orchestration maximize tool effectiveness

---

**Previous Module:** [Module 10: Database, Frameworks & Deployment](../module_10.md)  
**Next Module:** [Module 12: End-to-End Agentic AI System](../module_12.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

