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

#### ChromaDB Overview - Complete Analysis

ChromaDB is an open-source vector database designed specifically for AI applications, particularly for storing and retrieving embeddings efficiently. This section provides a comprehensive analysis of ChromaDB architecture, operations, and use cases.

**ChromaDB Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              CHROMADB ARCHITECTURE                            │
└─────────────────────────────────────────────────────────────┘

Client Application
    │
    ▼
┌──────────────────┐
│ ChromaDB Client  │
│ • Python API     │
│ • HTTP API       │
│ • Operations     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Collection       │
│ • Documents      │
│ • Embeddings     │
│ • Metadata       │
└────────┬─────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Embedding        │ │ Vector Storage  │ │ Metadata Index   │
│ Function         │ │ • Efficient     │ │ • Fast filtering │
│ • Generate       │ │ • Indexed       │ │ • Query support  │
│ • Store          │ │ • Retrieval     │ │ • Metadata ops   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

**What is ChromaDB? - Detailed Explanation:**

```
ChromaDB_Definition:

Type: Vector Database (Embedding Database)

Purpose:
Store and retrieve embeddings (vectors) efficiently
Enable similarity search
Support metadata filtering

Mathematical Model:
For collection C:
    C = {d_i, e_i, m_i}_{i=1}^N

Where:
- d_i: Document text
- e_i: Embedding vector (e_i ∈ R^d)
- m_i: Metadata dictionary
- N: Number of documents

Query Operation:
query(q, k) = Top_k({similarity(q, e_i) : e_i ∈ C})

Where:
- q: Query embedding
- k: Number of results
- similarity: Cosine similarity, Euclidean distance, etc.
```

**Key Features - Detailed Analysis:**

```
1. Lightweight and Fast:
   - Minimal dependencies
   - Fast in-memory operations
   - Efficient indexing
   - Low latency queries
   
   Performance Characteristics:
   - Query time: O(log N) with indexing
   - Insertion: O(1) amortized
   - Memory efficient: Optimized storage

2. Easy to Use:
   - Simple Python API
   - Minimal setup required
   - Intuitive operations
   - Good documentation

3. Persistent Storage:
   - Disk-based storage
   - Data persistence
   - Automatic saving
   - Recovery support

4. Metadata Filtering:
   - Filter by metadata
   - Complex queries
   - Efficient indexing
   - Fast filtering

5. Built-in Embedding Functions:
   - OpenAI embeddings
   - Sentence transformers
   - Custom functions
   - Easy integration
```

**Use Cases - Detailed Analysis:**

```
1. RAG Systems:
   - Store document embeddings
   - Retrieve relevant chunks
   - Support context retrieval
   - Enable hybrid search

2. Semantic Search:
   - Find similar documents
   - Query by meaning
   - Context-aware search
   - Multi-language support

3. Recommendation Systems:
   - User-item similarity
   - Content-based filtering
   - Collaborative filtering
   - Personalized recommendations

4. Similarity Search:
   - Find nearest neighbors
   - Clustering applications
   - Deduplication
   - Similarity matching
```

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

#### Ingestion Process - Complete Analysis

The ingestion process in ChromaDB involves creating collections, adding documents, generating embeddings, and storing them efficiently. This section provides a comprehensive analysis of the ingestion workflow.

**Ingestion Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              CHROMADB INGESTION PROCESS                       │
└─────────────────────────────────────────────────────────────┘

Documents (Text)
    │
    ▼
┌──────────────────┐
│ Step 1: Create   │
│ Collection       │
│ • Define name    │
│ • Set embedding │
│   function       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Step 2: Add      │
│ Documents        │
│ • Text           │
│ • IDs            │
│ • Metadata       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Step 3: Generate │
│ Embeddings       │
│ • Auto-embed     │
│ • Or use custom  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Step 4: Store    │
│ • Embeddings     │
│ • Metadata       │
│ • Index          │
└──────────────────┘
```

**Step 1: Create Collection - Detailed:**

```
Collection_Creation:

Collection Structure:
Collection = {
    name: str,
    embedding_function: Callable,
    metadata: dict,
    documents: List[str],
    embeddings: np.ndarray,
    metadatas: List[dict],
    ids: List[str]
}

Mathematical Model:
C = Collection(name, embedding_fn, metadata)

Where:
- name: Unique collection identifier
- embedding_fn: Function f: text → R^d
- metadata: Collection-level metadata

Operations:
1. Create: C = create_collection(name, embedding_fn)
2. Get: C = get_collection(name)
3. List: collections = list_collections()
4. Delete: delete_collection(name)
```

**Step 2: Add Documents - Mathematical Model:**

```
Document_Addition:

Input:
- Documents: D = [d₁, d₂, ..., dₙ]
- IDs: I = [id₁, id₂, ..., idₙ]
- Metadatas: M = [m₁, m₂, ..., mₙ]

Process:
1. Validate inputs
2. Generate embeddings: E = [embedding_fn(d_i) for d_i in D]
3. Store: Collection.add(documents=D, ids=I, metadatas=M)

Mathematical Model:
For each document d_i:
    e_i = embedding_fn(d_i)  # e_i ∈ R^d
    Store(d_i, e_i, id_i, m_i)

Where:
- d: Embedding dimension (e.g., 384, 768, 1536)
- e_i: Embedding vector
- id_i: Unique identifier
- m_i: Metadata dictionary

Batch Operations:
add_batch = {
    documents: [d₁, d₂, ..., dₙ],
    ids: [id₁, id₂, ..., idₙ],
    metadatas: [m₁, m₂, ..., mₙ]
}

Efficiency:
- Batch processing: O(n) for n documents
- Parallel embedding generation
- Optimized storage
```

**Step 3: Embedding Generation - Complete Analysis:**

```
Embedding_Generation:

Automatic Generation:
When adding documents without embeddings:
    E = [embedding_fn(d) for d in documents]

Where:
- embedding_fn: Collection's embedding function
- E: List of embedding vectors

Custom Embeddings:
Can provide pre-computed embeddings:
    collection.add(
        documents=documents,
        embeddings=embeddings,  # Pre-computed
        ids=ids,
        metadatas=metadatas
    )

Embedding Function Types:
1. OpenAI Embeddings:
   embedding_fn = OpenAIEmbeddingFunction(
       api_key=api_key,
       model_name="text-embedding-ada-002"
   )
   Dimension: 1536

2. Sentence Transformers:
   embedding_fn = SentenceTransformerEmbeddingFunction(
       model_name="all-MiniLM-L6-v2"
   )
   Dimension: 384

3. Custom Function:
   def custom_embedding(texts):
       # Your embedding logic
       return embeddings
   
   embedding_fn = custom_embedding
```

**Storage Optimization:**

```
Storage_Optimization:

Embedding Storage:
- Efficient vector storage
- Compression (optional)
- Indexed for fast retrieval
- Memory-mapped files

Metadata Storage:
- Indexed metadata
- Fast filtering
- Query optimization
- Efficient lookups

Index Structure:
- HNSW index for vectors
- B-tree for metadata
- Optimized for queries
- Fast similarity search

Storage Format:
- Columnar storage
- Efficient serialization
- Fast deserialization
- Memory efficient
```

**Complete Ingestion Implementation:**

```python
"""
Complete ChromaDB Ingestion Implementation
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import numpy as np

class ChromaDBIngestion:
    """
    Complete ChromaDB ingestion system.
    
    Mathematical Model:
        Collection = {documents, embeddings, metadatas, ids}
        embedding: text → R^d
    """
    
    def __init__(self, 
                 collection_name: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            collection_name: Name of collection
            embedding_model: Embedding model name
            persist_directory: Directory for persistence
        """
        # Initialize client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
            print(f"[ChromaDB] Persistent client initialized: {persist_directory}")
        else:
            self.client = chromadb.Client()
            print("[ChromaDB] In-memory client initialized")
        
        # Setup embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        print(f"[ChromaDB] Embedding model: {embedding_model}")
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            print(f"[ChromaDB] Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            print(f"[ChromaDB] Created new collection: {collection_name}")
    
    def add_documents(self,
                     documents: List[str],
                     ids: Optional[List[str]] = None,
                     metadatas: Optional[List[Dict]] = None):
        """
        Add documents to collection.
        
        Mathematical Model:
            For each document d_i:
                e_i = embedding_fn(d_i)
                Store(d_i, e_i, id_i, m_i)
        
        Args:
            documents: List of document texts
            ids: Optional list of IDs
            metadatas: Optional list of metadata dictionaries
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        
        n_docs = len(documents)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(n_docs)]
        
        # Generate metadata if not provided
        if metadatas is None:
            metadatas = [{}] * n_docs
        
        # Validate lengths
        if len(ids) != n_docs or len(metadatas) != n_docs:
            raise ValueError("Lengths of documents, ids, and metadatas must match")
        
        print(f"[Ingestion] Adding {n_docs} documents...")
        
        # Add to collection
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"[Ingestion] Successfully added {n_docs} documents")
        print(f"[Ingestion] Collection size: {self.collection.count()}")
    
    def add_documents_batch(self,
                           batches: List[Dict],
                           batch_size: int = 100):
        """
        Add documents in batches for efficiency.
        
        Args:
            batches: List of batch dictionaries
            batch_size: Size of each batch
        """
        total_docs = sum(len(batch['documents']) for batch in batches)
        print(f"[Batch Ingestion] Processing {total_docs} documents in batches...")
        
        for i, batch in enumerate(batches):
            print(f"[Batch Ingestion] Processing batch {i+1}/{len(batches)}")
            self.add_documents(
                documents=batch['documents'],
                ids=batch.get('ids'),
                metadatas=batch.get('metadatas')
            )
        
        print(f"[Batch Ingestion] Completed! Total documents: {self.collection.count()}")
    
    def get_collection_info(self) -> Dict:
        """
        Get collection information.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        return {
            'name': self.collection.name,
            'count': count,
            'embedding_dimension': len(self.embedding_fn(["test"])[0])
        }


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("CHROMADB INGESTION - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    # Initialize
    ingestion = ChromaDBIngestion(
        collection_name="test_collection",
        embedding_model="all-MiniLM-L6-v2",
        persist_directory="./chroma_db"
    )
    
    # Add documents
    documents = [
        "Artificial intelligence is transforming industries.",
        "Machine learning enables data-driven decisions.",
        "Deep learning powers modern AI applications."
    ]
    
    metadatas = [
        {"source": "doc1", "topic": "AI"},
        {"source": "doc2", "topic": "ML"},
        {"source": "doc3", "topic": "DL"}
    ]
    
    ids = ["doc1", "doc2", "doc3"]
    
    ingestion.add_documents(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    # Get info
    info = ingestion.get_collection_info()
    print(f"\nCollection Info: {info}")
    
    print("\n" + "=" * 70)
    print("INGESTION DEMO COMPLETE")
    print("=" * 70)

"""
Expected Output:
======================================================================
CHROMADB INGESTION - COMPLETE IMPLEMENTATION
======================================================================
[ChromaDB] Persistent client initialized: ./chroma_db
[ChromaDB] Embedding model: all-MiniLM-L6-v2
[ChromaDB] Created new collection: test_collection
[Ingestion] Adding 3 documents...
[Ingestion] Successfully added 3 documents
[Ingestion] Collection size: 3

Collection Info: {'name': 'test_collection', 'count': 3, 'embedding_dimension': 384}

======================================================================
INGESTION DEMO COMPLETE
======================================================================
"""
```

#### Querying - Complete Mathematical Analysis

Querying in ChromaDB involves similarity search, metadata filtering, and retrieval operations. This section provides a comprehensive analysis of query operations and their mathematical foundations.

**Query Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              CHROMADB QUERY ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

Query Text
    │
    ▼
┌──────────────────┐
│ Embedding        │
│ Generation       │
│ q = embed(query) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Similarity       │
│ Search           │
│ • Vector search  │
│ • Index lookup   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Metadata         │
│ Filtering        │
│ • Filter results │
│ • Apply filters  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Top-K Results    │
│ • Ranked         │
│ • Filtered       │
└──────────────────┘
```

**Similarity Search - Mathematical Model:**

```
Similarity_Search_Model:

Query:
q = embedding_fn(query_text)  # q ∈ R^d

Collection:
C = {(d_i, e_i, m_i)}_{i=1}^N

Where:
- d_i: Document text
- e_i: Embedding vector
- m_i: Metadata

Similarity Computation:
For each document i:
    similarity_i = similarity(q, e_i)

Where similarity can be:
1. Cosine Similarity:
   similarity(q, e) = (q · e) / (||q|| × ||e||)
   
2. Euclidean Distance:
   distance(q, e) = ||q - e||
   similarity(q, e) = 1 / (1 + distance(q, e))

3. Dot Product:
   similarity(q, e) = q · e

Top-K Selection:
results = Top_k({(d_i, similarity_i, m_i) : i = 1 to N})

Where:
- Top_k: Select k highest similarity scores
- results: List of (document, similarity, metadata) tuples

Query Operation:
query(query_text, k) = Top_k(Similarity_Search(q, C))

Where:
- q = embedding_fn(query_text)
- k: Number of results
```

**Metadata Filtering - Complete Analysis:**

```
Metadata_Filtering_Model:

Filter Condition:
F = {key: value, ...}

Filtered Collection:
C_filtered = {(d_i, e_i, m_i) : m_i satisfies F}

Where:
- m_i satisfies F if all key-value pairs in F match m_i

Query with Filtering:
query(query_text, k, filter=F) = Top_k(Similarity_Search(q, C_filtered))

Filter Operations:
1. Equality: {"source": "doc1"}
2. Range: {"date": {"$gte": "2023-01-01"}}
3. In: {"category": {"$in": ["AI", "ML"]}}
4. And/Or: {"$and": [{"source": "doc1"}, {"topic": "AI"}]}

Example:
Query: "What is AI?"
Filter: {"source": "doc1", "topic": "AI"}
Results: Top-5 documents matching filter with highest similarity
```

**Complete Query Implementation:**

```python
"""
Complete ChromaDB Query Implementation
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Union
import numpy as np

class ChromaDBQuery:
    """
    Complete ChromaDB query system.
    
    Mathematical Model:
        query(q, k, filter) = Top_k(Similarity_Search(q, C_filtered))
    """
    
    def __init__(self, collection):
        """
        Initialize query system.
        
        Args:
            collection: ChromaDB collection object
        """
        self.collection = collection
        print("[Query] Initialized query system")
    
    def similarity_search(self,
                         query_text: str,
                         n_results: int = 5,
                         where: Optional[Dict] = None) -> Dict:
        """
        Perform similarity search.
        
        Mathematical Model:
            q = embedding_fn(query_text)
            results = Top_k(Similarity_Search(q, C_filtered))
        
        Args:
            query_text: Query text
            n_results: Number of results
            where: Metadata filter
            
        Returns:
            Dictionary with documents, distances, metadatas
        """
        print(f"[Query] Searching: '{query_text}'")
        print(f"[Query] Results requested: {n_results}")
        if where:
            print(f"[Query] Filter: {where}")
        
        # Perform query
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        print(f"[Query] Found {len(results['ids'][0])} results")
        
        return results
    
    def get_by_ids(self, ids: List[str]) -> Dict:
        """
        Get documents by IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            Dictionary with documents, embeddings, metadatas
        """
        print(f"[Query] Getting documents by IDs: {ids}")
        
        results = self.collection.get(ids=ids)
        
        print(f"[Query] Retrieved {len(results['ids'])} documents")
        
        return results
    
    def hybrid_search(self,
                     query_text: str,
                     n_results: int = 5,
                     where: Optional[Dict] = None,
                     include: List[str] = ["documents", "metadatas", "distances"]) -> Dict:
        """
        Perform hybrid search with metadata filtering.
        
        Args:
            query_text: Query text
            n_results: Number of results
            where: Metadata filter
            include: What to include in results
            
        Returns:
            Dictionary with query results
        """
        print(f"[Hybrid Search] Query: '{query_text}'")
        print(f"[Hybrid Search] Filter: {where}")
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=include
        )
        
        # Format results
        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i] if 'documents' in results else None,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else None,
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
        
        print(f"[Hybrid Search] Found {len(formatted_results)} results")
        
        return {
            'results': formatted_results,
            'raw': results
        }


# Example Usage
if __name__ == "__main__":
    print("=" * 70)
    print("CHROMADB QUERY - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    # Initialize (assuming collection exists)
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(
        name="test_collection",
        embedding_function=embedding_fn
    )
    
    query_system = ChromaDBQuery(collection)
    
    # Similarity search
    results = query_system.similarity_search(
        query_text="What is artificial intelligence?",
        n_results=3
    )
    
    print("\nSearch Results:")
    for i, (doc_id, doc, metadata) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0]
    )):
        print(f"\nResult {i+1}:")
        print(f"ID: {doc_id}")
        print(f"Document: {doc[:100]}...")
        print(f"Metadata: {metadata}")
        print(f"Distance: {results['distances'][0][i]:.4f}")
    
    # Filtered search
    filtered_results = query_system.similarity_search(
        query_text="machine learning",
        n_results=2,
        where={"topic": "ML"}
    )
    
    print("\n" + "=" * 70)
    print("QUERY DEMO COMPLETE")
    print("=" * 70)

"""
Expected Output:
======================================================================
CHROMADB QUERY - COMPLETE IMPLEMENTATION
======================================================================
[Query] Initialized query system
[Query] Searching: 'What is artificial intelligence?'
[Query] Results requested: 3
[Query] Found 3 results

Search Results:

Result 1:
ID: doc1
Document: Artificial intelligence is transforming industries....
Metadata: {'source': 'doc1', 'topic': 'AI'}
Distance: 0.2341

Result 2:
ID: doc2
Document: Machine learning enables data-driven decisions....
Metadata: {'source': 'doc2', 'topic': 'ML'}
Distance: 0.4567

Result 3:
ID: doc3
Document: Deep learning powers modern AI applications....
Metadata: {'source': 'doc3', 'topic': 'DL'}
Distance: 0.5678

======================================================================
QUERY DEMO COMPLETE
======================================================================
"""
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

#### API Deployment with FastAPI - Complete Analysis

FastAPI is a modern web framework for building APIs with Python. It's particularly well-suited for LLM and RAG applications due to its async support, automatic documentation, and type safety.

**FastAPI Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              FASTAPI DEPLOYMENT ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────┘

Client Request
    │
    ▼
┌──────────────────┐
│ FastAPI Server   │
│ • Routing        │
│ • Validation     │
│ • Middleware     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Request Handler  │
│ • Parse input    │
│ • Validate       │
│ • Process        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ RAG Pipeline     │
│ • Retrieve       │
│ • Generate       │
│ • Format         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Response         │
│ • JSON format    │
│ • Error handling │
└──────────────────┘
```

**Why FastAPI? - Detailed Analysis:**

```
FastAPI_Advantages:

1. Performance:
   - Fast (comparable to Node.js)
   - Async support
   - High concurrency
   - Efficient for I/O-bound tasks
   
   Benchmark:
   - FastAPI: ~50k req/s
   - Flask: ~20k req/s
   - Django: ~15k req/s

2. Automatic Documentation:
   - OpenAPI (Swagger) UI
   - ReDoc documentation
   - Automatic schema generation
   - Interactive API testing

3. Type Safety:
   - Pydantic models
   - Type hints
   - Automatic validation
   - Better IDE support

4. Async Support:
   - Native async/await
   - Non-blocking I/O
   - Better concurrency
   - Perfect for LLM APIs

5. Modern Features:
   - WebSocket support
   - Background tasks
   - Dependency injection
   - Easy testing
```

**Complete FastAPI RAG Deployment:**

```python
"""
Complete FastAPI RAG Deployment
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os
import time
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., description="User question", min_length=1)
    max_results: int = Field(default=5, description="Maximum retrieval results", ge=1, le=20)
    temperature: float = Field(default=0.0, description="LLM temperature", ge=0.0, le=2.0)
    include_sources: bool = Field(default=True, description="Include source documents")

class SourceDocument(BaseModel):
    """Source document model."""
    content: str
    metadata: Dict
    score: Optional[float] = None

class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[SourceDocument] = []
    query_time: float
    timestamp: str

# Global RAG components
vectorstore = None
qa_chain = None

def initialize_rag():
    """Initialize RAG components."""
    global vectorstore, qa_chain
    
    logger.info("[Initialization] Loading vector store...")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings()
    )
    
    logger.info("[Initialization] Creating QA chain...")
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    logger.info("[Initialization] RAG system ready!")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    initialize_rag()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Mathematical Model:
        answer = RAG(query, context)
        where context = Retrieve(query, vectorstore)
    
    Args:
        request: Query request
        
    Returns:
        Query response with answer and sources
    """
    start_time = time.time()
    
    try:
        logger.info(f"[Query] Processing: '{request.question}'")
        
        # Update retriever if needed
        if request.max_results != 5:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": request.max_results}
            )
            qa_chain.retriever = retriever
        
        # Execute query
        result = qa_chain({
            "query": request.question
        })
        
        # Extract answer
        answer = result["result"]
        
        # Extract sources
        sources = []
        if request.include_sources and "source_documents" in result:
            for doc in result["source_documents"]:
                source = SourceDocument(
                    content=doc.page_content[:500],  # Truncate for response
                    metadata=doc.metadata,
                    score=None  # Could add similarity score
                )
                sources.append(source)
        
        query_time = time.time() - start_time
        
        logger.info(f"[Query] Completed in {query_time:.2f}s")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query_time=query_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"[Query] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vectorstore_ready": vectorstore is not None,
        "qa_chain_ready": qa_chain is not None
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if vectorstore:
        collection = vectorstore._collection
        count = collection.count()
        return {
            "collection_count": count,
            "vectorstore_ready": True
        }
    return {"vectorstore_ready": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

"""
Expected Output:
======================================================================
RAG API - COMPLETE IMPLEMENTATION
======================================================================

[Initialization] Loading vector store...
[Initialization] Creating QA chain...
[Initialization] RAG system ready!

API Documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Example Request:
POST http://localhost:8000/query
{
    "question": "What is AI?",
    "max_results": 5,
    "include_sources": true
}

Example Response:
{
    "answer": "AI is artificial intelligence...",
    "sources": [...],
    "query_time": 1.23,
    "timestamp": "2024-01-01T12:00:00"
}
"""
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

