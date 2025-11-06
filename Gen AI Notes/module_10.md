# Module 10: Database, Frameworks & Deployment

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Classes:** 17-18

---

## Class 17: ChromaDB (Vector Database) – Full Notes

### Topics Covered

- Architecture, ingestion, querying, filtering
- Integration with LangChain / LlamaIndex
- Comparison with Pinecone, Weaviate, Milvus
- Production deployment considerations

### Learning Objectives

By the end of this class, students will be able to:
- Understand vector database architecture
- Use ChromaDB for RAG applications
- Integrate ChromaDB with frameworks
- Compare different vector databases
- Deploy vector databases in production

### Core Concepts

#### ChromaDB Overview

**What is ChromaDB?**
- Open-source vector database
- Designed for AI applications
- Simple Python API
- Embedding storage and retrieval

**Key Features:**
- Lightweight and fast
- Easy to use
- Persistent storage
- Metadata filtering
- Built-in embedding functions

**Use Cases:**
- RAG systems
- Semantic search
- Recommendation systems
- Similarity search

#### Architecture

**Core Components:**

**1. Collection:**
- Group of embeddings
- Similar to database table
- Contains documents, embeddings, metadata

**2. Embedding Function:**
- Converts text to vectors
- Built-in or custom
- Examples: OpenAI, Sentence Transformers

**3. Storage:**
- In-memory (default)
- Persistent (disk-based)
- Client-server mode

**4. Query Interface:**
- Similarity search
- Metadata filtering
- Hybrid queries

#### Ingestion Process

**Step 1: Create Collection**
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)
```

**Step 2: Add Documents**
```python
collection.add(
    documents=["Document 1", "Document 2"],
    ids=["id1", "id2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)
```

**Step 3: Embedding Generation**
- Automatic with embedding function
- Or provide pre-computed embeddings

**Storage:**
- Embeddings stored efficiently
- Metadata indexed
- Fast retrieval

#### Querying

**Similarity Search:**
```python
results = collection.query(
    query_texts=["What is AI?"],
    n_results=5
)
```

**With Metadata Filtering:**
```python
results = collection.query(
    query_texts=["What is AI?"],
    n_results=5,
    where={"source": "doc1"}
)
```

**Get by IDs:**
```python
results = collection.get(
    ids=["id1", "id2"]
)
```

**Update and Delete:**
```python
collection.update(
    ids=["id1"],
    documents=["Updated document"],
    metadatas=[{"source": "updated"}]
)

collection.delete(ids=["id1"])
```

#### Integration with LangChain

**Basic Integration:**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

**Retriever:**
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
```

**With Metadata:**
```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_collection",
    persist_directory="./chroma_db"
)
```

#### Integration with LlamaIndex

**Basic Integration:**
```python
from llama_index import VectorStoreIndex, StorageContext
from llama_index.vector_stores import ChromaVectorStore
import chromadb

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("my_collection")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

#### Comparison with Other Vector Databases

**ChromaDB vs Pinecone:**

| Feature | ChromaDB | Pinecone |
|---------|----------|----------|
| Type | Open-source | Managed service |
| Setup | Simple | Very simple |
| Scalability | Limited | Very high |
| Cost | Free | Paid |
| Hosting | Self-hosted | Cloud |
| Best For | Development, small-scale | Production, large-scale |

**ChromaDB vs Weaviate:**

| Feature | ChromaDB | Weaviate |
|---------|----------|----------|
| Simplicity | Very simple | Moderate |
| Features | Basic | Advanced |
| GraphQL | No | Yes |
| Vectorization | Optional | Built-in |
| Use Cases | Simple RAG | Complex search |

**ChromaDB vs Milvus:**

| Feature | ChromaDB | Milvus |
|---------|----------|--------|
| Complexity | Simple | Complex |
| Scalability | Limited | Very high |
| Features | Basic | Advanced |
| Best For | Small-medium | Enterprise |

**When to Use ChromaDB:**
- Development and prototyping
- Small to medium datasets
- Simple use cases
- Cost-sensitive projects
- Local deployment

**When to Use Alternatives:**
- **Pinecone:** Production, large-scale, managed service
- **Weaviate:** Complex queries, graph features
- **Milvus:** Enterprise, very large scale
- **FAISS:** Research, in-memory, library

#### Production Deployment

**Considerations:**

**1. Persistence:**
```python
client = chromadb.PersistentClient(path="./chroma_db")
```

**2. Client-Server Mode:**
```python
# Server
chroma run --path ./chroma_db

# Client
client = chromadb.HttpClient(host="localhost", port=8000)
```

**3. Scaling:**
- Horizontal scaling limited
- Consider alternatives for large scale
- Sharding strategies

**4. Monitoring:**
- Collection size
- Query performance
- Memory usage

**5. Backup:**
- Regular backups
- Data persistence
- Recovery strategies

---

## Class 18: Model Deployment with Flask / FastAPI

### Topics Covered

- Exposing LLM or RAG as API endpoints
- Building interactive UIs with Gradio / Streamlit
- Integration with MLflow for version control
- Production deployment best practices

### Learning Objectives

By the end of this class, students will be able to:
- Deploy LLM applications as APIs
- Create user interfaces for GenAI apps
- Implement model versioning
- Deploy applications in production
- Monitor and maintain deployed systems

### Core Concepts

#### API Deployment with FastAPI

**Why FastAPI?**
- Fast and modern
- Automatic API documentation
- Type hints support
- Async support
- Easy to use

**Basic FastAPI Setup:**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    context: str = None

class QueryResponse(BaseModel):
    answer: str
    sources: list = []

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # Your RAG logic here
    answer = rag_pipeline(request.question)
    return QueryResponse(answer=answer, sources=[])
```

**RAG API Example:**
```python
from fastapi import FastAPI
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

app = FastAPI()

# Initialize components
vectorstore = Chroma(persist_directory="./chroma_db")
llm = OpenAI()

@app.post("/rag")
async def rag_query(request: QueryRequest):
    # Retrieve context
    docs = vectorstore.similarity_search(request.question, k=4)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Generate answer
    prompt = f"Context: {context}\nQuestion: {request.question}\nAnswer:"
    answer = llm(prompt)
    
    return {"answer": answer, "sources": [doc.metadata for doc in docs]}
```

**API Features:**
- Request validation
- Error handling
- Rate limiting
- Authentication
- Logging

#### Deployment with Flask

**Basic Flask Setup:**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    
    # Your RAG logic
    answer = rag_pipeline(question)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Comparison:**
- **FastAPI:** Modern, async, auto-docs, better performance
- **Flask:** Simple, flexible, mature, larger ecosystem

#### Interactive UIs with Gradio

**What is Gradio?**
- Quick UI for ML models
- Python-based
- Shareable interfaces
- No frontend knowledge needed

**Basic Gradio Interface:**
```python
import gradio as gr

def rag_interface(question):
    answer = rag_pipeline(question)
    return answer

interface = gr.Interface(
    fn=rag_interface,
    inputs="textbox",
    outputs="textbox",
    title="RAG Question Answering",
    description="Ask questions about your documents"
)

interface.launch()
```

**Advanced Features:**
- Multiple inputs/outputs
- File uploads
- Chat interface
- Custom components
- Sharing capabilities

**Chat Interface:**
```python
import gradio as gr

def chat(message, history):
    response = rag_pipeline(message)
    return response

demo = gr.ChatInterface(
    fn=chat,
    title="RAG Chatbot",
    examples=["What is AI?", "How does RAG work?"]
)

demo.launch()
```

#### Interactive UIs with Streamlit

**What is Streamlit?**
- Python-based web framework
- Easy to build apps
- Interactive widgets
- Good for data apps

**Basic Streamlit App:**
```python
import streamlit as st

st.title("RAG Question Answering")

question = st.text_input("Enter your question:")

if question:
    answer = rag_pipeline(question)
    st.write("Answer:", answer)
    st.write("Sources:", sources)
```

**Advanced Features:**
- Sidebar navigation
- File uploads
- Session state
- Caching
- Custom components

**RAG App Example:**
```python
import streamlit as st

st.set_page_config(page_title="RAG System")

st.title("Document Q&A System")

# File upload
uploaded_file = st.file_uploader("Upload document", type=["pdf", "txt"])

if uploaded_file:
    # Process document
    documents = process_document(uploaded_file)
    st.success("Document processed!")
    
    # Question input
    question = st.text_input("Ask a question:")
    
    if question:
        # Get answer
        answer = rag_pipeline(question, documents)
        st.write("**Answer:**", answer)
```

#### MLflow Integration

**What is MLflow?**
- ML lifecycle management
- Experiment tracking
- Model versioning
- Model registry

**Tracking Experiments:**
```python
import mlflow

mlflow.set_experiment("rag_experiments")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "gpt-3.5-turbo")
    mlflow.log_param("retrieval_k", 4)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("latency", 0.5)
    
    # Log model
    mlflow.pyfunc.log_model("rag_model", python_model=rag_pipeline)
```

**Model Registry:**
```python
# Register model
model_version = mlflow.register_model(
    model_uri="runs:/run_id/rag_model",
    name="ProductionRAG"
)

# Load model
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/ProductionRAG/{model_version.version}"
)
```

**Benefits:**
- Version control
- Reproducibility
- Experiment comparison
- Model management

#### Production Deployment

**Considerations:**

**1. Performance:**
- Async processing
- Caching
- Load balancing
- Connection pooling

**2. Scalability:**
- Horizontal scaling
- Containerization (Docker)
- Kubernetes for orchestration
- Auto-scaling

**3. Monitoring:**
- Logging
- Metrics collection
- Error tracking
- Performance monitoring

**4. Security:**
- Authentication
- API keys
- Rate limiting
- Input validation

**5. Reliability:**
- Error handling
- Retry logic
- Health checks
- Graceful degradation

**Docker Deployment:**
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Cloud Deployment Options:**
- **AWS:** EC2, Lambda, ECS, SageMaker
- **GCP:** Cloud Run, GKE, Vertex AI
- **Azure:** Container Instances, AKS, Azure ML
- **Heroku:** Simple PaaS
- **Render:** Easy deployment

### Readings

- Vector database documentation:
  - [ChromaDB Documentation](https://docs.trychroma.com/)
  - [Pinecone Documentation](https://docs.pinecone.io/)
  - [Weaviate Documentation](https://weaviate.io/developers/weaviate)

- API deployment best practices:
  - FastAPI documentation
  - Flask documentation
  - MLflow documentation

 

### Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)

### Practical Code Examples

#### Complete FastAPI RAG Deployment

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import os
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

app = FastAPI(title="RAG API", version="1.0.0")

# Initialize components
vectorstore = Chroma(persist_directory="./chroma_db")
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

class QueryRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 500

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system"""
    try:
        result = qa_chain({"query": request.question})
        
        sources = [
            {
                "content": doc.page_content[:200],
                "metadata": doc.metadata
            }
            for doc in result.get("source_documents", [])
        ]
        
        return QueryResponse(
            answer=result["result"],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Pro Tip:** Always include health check endpoints and error handling in production APIs. Use async functions for better performance with I/O operations.

**Common Pitfall:** Not implementing rate limiting can lead to API abuse and high costs. Always add rate limiting and authentication for production.

#### Gradio Chat Interface

```python
import gradio as gr
from langchain.chains import RetrievalQA

def create_gradio_interface(qa_chain):
    """Create Gradio interface for RAG"""
    
    def chat(message, history):
        """Chat function"""
        result = qa_chain({"query": message})
        return result["result"]
    
    interface = gr.ChatInterface(
        fn=chat,
        title="RAG Question Answering",
        description="Ask questions about your documents",
        examples=[
            "What is the main topic?",
            "Summarize the key points",
            "What are the key findings?"
        ],
        theme=gr.themes.Soft()
    )
    
    return interface

# Usage
# interface = create_gradio_interface(qa_chain)
# interface.launch(share=True)
```

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **API errors** | 500 errors, crashes | Add error handling, validate inputs, check logs |
| **Slow responses** | High latency | Optimize retrieval, use caching, async processing |
| **Memory issues** | Out of memory | Optimize vector store, use streaming, reduce batch size |
| **Deployment failures** | Container errors | Check Dockerfile, verify dependencies, check logs |
| **Vector DB connection** | Connection errors | Verify persistence path, check permissions, test locally |

### Quick Reference Guide

#### Deployment Stack Comparison

| Component | Development | Production |
|-----------|-------------|------------|
| Vector DB | ChromaDB | Pinecone/Milvus |
| API Framework | FastAPI/Flask | FastAPI + Gunicorn |
| UI | Gradio/Streamlit | React/Vue + API |
| Container | Docker | Kubernetes |
| Monitoring | Logs | Prometheus + Grafana |

### Case Studies

#### Case Study: Production RAG API Deployment

**Challenge:** Deploy RAG system for 1000+ daily users.

**Solution:**
- FastAPI with async processing
- Pinecone for vector storage
- Docker containers on Kubernetes
- MLflow for model versioning

**Results:**
- 99.9% uptime
- <2 second response time
- Handles 100+ concurrent requests

**Lessons Learned:**
- Async processing critical for scalability
- Proper monitoring essential
- Container orchestration enables scaling

### Key Takeaways

1. ChromaDB is excellent for development and small-scale deployments
2. FastAPI provides modern, fast API development
3. Gradio and Streamlit enable rapid UI development
4. MLflow helps manage ML lifecycle and experiments
5. Production deployment requires careful consideration of scalability and reliability
6. Vector database choice depends on scale and requirements
7. Proper deployment practices ensure system reliability and maintainability
8. Async processing and caching improve performance
9. Monitoring and logging are essential for production
10. Containerization enables consistent deployments

---

**Previous Module:** [Module 9: LLM Inference & Prompt Engineering](../module_09.md)  
**Next Module:** [Module 11: Frameworks, Libraries & Platforms Overview](../module_11.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

