# 01 — Introduction to ChromaDB

## Overview

ChromaDB is a lightweight, developer-friendly vector database designed to store, index, and query high-dimensional embeddings for semantic search and retrieval-augmented generation (RAG) workflows. It aims to provide a straightforward API for ingesting vector embeddings alongside structured metadata and text content, enabling efficient nearest-neighbor search and filtered queries.

This chapter provides a comprehensive introduction: conceptual foundations, core use cases, architectural overview, feature breakdown, comparisons with other vector stores, practical examples, and decision guidance for when to adopt ChromaDB.

---

## Why vector databases exist

Traditional databases and search engines rely on lexical matching (keywords). Modern NLP applications require semantic understanding — matching meaning rather than exact words. Vector databases store numeric representations (embeddings) of text, images, or other modalities and provide efficient nearest-neighbor search over these vectors to find semantically similar items.

Key motivations:

- Move beyond keyword matching to semantic similarity.
- Support retrieval for LLM contexts (RAG) to provide relevant, grounded information.
- Combine metadata filtering with semantic ranking for precise, contextual retrieval.

---

## Core use cases

- Retrieval-Augmented Generation (RAG): fetch the most relevant documents or passages to use as context for an LLM prompt.
- Semantic search: allow natural-language queries to return relevant documents even when keywords differ.
- Question answering over a knowledge base: match user questions to best supporting passages.
- Recommendation systems: find similar items based on content embeddings.
- Chatbots and conversational agents: surface relevant context snippets from conversation history or knowledge stores.

---

## Key features and capabilities

- Vector similarity search (cosine / dot-product / euclidean) with tunable result counts.
- Collections: logical grouping of vectors and metadata (similar to tables or indices).
- Simple client APIs for adding/updating/deleting vectors and metadata.
- Support for metadata filters to narrow candidate sets before or after vector scoring.
- Persistence options: in-memory for prototyping, on-disk backends for durability.
- Lightweight deployment suitable for local, embedded, or containerized production use.

---

## High-level architecture

At a high level, a ChromaDB deployment consists of:

- Client(s): applications that create embeddings, push them to Chroma, and query.
- Collections: named groupings of items (id, embedding, metadata, optional text).
- Index layer: the internal vector index implementation used for approximate nearest neighbor (ANN) or exact search.
- Storage backend: memory, local disk, or external storage integration for persistence.

Flow (ingest + query):

1. Documents are chunked and associated with metadata.
2. An embedder (OpenAI, Hugging Face, etc.) produces an embedding vector for each chunk.
3. The client inserts items into a Chroma collection.
4. For a query, an embedding is produced for the query text and used to query the collection.
5. Optionally, metadata filters are applied to restrict the search domain, then results are reranked.

Mermaid flow diagram:

```mermaid
flowchart LR
  A[Source Documents] --> B[Chunking]
  B --> C[Embedding Model]
  C --> D[Chroma Collection (add)]
  E[User Query] --> F[Query Embedding]
  F --> G[Chroma Collection (query)]
  G --> H[Top-k Results]
  H --> I[Optional Rerank & LLM Context]
```

---

## Comparison with other vector stores (quick reference)

| Capability | ChromaDB | FAISS | Milvus | Pinecone |
|---|---:|---:|---:|---:|
| Ease of use / API | High | Medium | Medium | High |
| Managed service | No (self-host) | No (library) | Yes/No | Yes |
| Persistence options | In-memory / disk | Library-managed | Server-based | Managed |
| Scaling (multi-node) | Limited / depends on deployment | Limited | Good | Excellent |
| Integrations | Embedders, Python | Library-only | Connectors | Managed SDKs |

Notes: The right choice depends on scale, operations, and whether you prefer managed services.

---

## When to choose ChromaDB

- Prototyping and local development where simplicity matters.
- Small-to-medium production workloads where a lightweight, embeddable vector store is acceptable.
- Use cases that require tight control over storage and index behavior without a managed service.

Consider alternatives when you need large-scale, multi-node sharding with high availability and sophisticated cluster management — then consider Milvus, or managed offerings like Pinecone or a hosted vector DB.

---

## In-depth conceptual topics

### Collections

- A `collection` is a namespaced container for items. Each item typically has:
  - `id` (string)
  - `embedding` (float array)
  - `metadata` (dictionary of fields)
  - `document` (optional text)

- Collections enable logical separation: store different datasets (e.g., `docs`, `conversations`, `products`) in separate collections to control indexing and metadata schema.

### Items and metadata

- Metadata fields should be small, typed (string/number/bool), and used for filtering and faceting.
- Best practice: include `source`, `created_at`, `doc_id`, and `chunk_index` for traceability.

### Indexing and similarity metrics

- Chroma uses internal index structures; depending on implementation and backend, ANN algorithms (HNSW, IVF, PQ) may be used.
- Select similarity metric according to embedding semantics:
  - Cosine (most common for semantic text embeddings)
  - Dot product (when magnitude contains meaning)
  - Euclidean (L2) for dense vector spaces where distance indicates dissimilarity

---

## Practical examples (conceptual)

### Example 1 — Basic Python quickstart (conceptual)

```py
from chromadb import Client

client = Client()
col = client.create_collection('articles')

# Add one document (idempotent upsert by id)
col.add(
    ids=['article-001'],
    embeddings=[[0.12, -0.34, 0.98, ...]],
    metadatas=[{'source':'web','author':'Alice'}],
    documents=['Full article text or summary...']
)

# Query with an embedding
results = col.query(query_embeddings=[[0.05, -0.2, 0.9, ...]], n_results=5)
```

### Example 2 — RAG flow (high-level)

1. Receive user question.
2. Create embedding for question.
3. Query Chroma collection for top-k relevant chunks.
4. Concatenate top chunks as context and pass to LLM with prompt template.
5. Return LLM response and optionally store the QA exchange in a `conversations` collection.

---

## Design considerations and trade-offs

- Chunk size: smaller chunks yield more precise matches but increase index size and potentially harm coherence; larger chunks keep context but may reduce recall for fine-grained queries.
- Embedding model choice: higher-quality models cost more; choose model balancing precision and cost.
- Normalization: ensure consistent vector normalization if using cosine similarity.

Table: Chunking trade-offs

| Chunk size | Pros | Cons | Typical use |
|---:|---|---|---|
| Small (50–200 tokens) | Fine-grained retrieval, better pinpointing | More vectors, larger index | QA over long docs |
| Medium (200–500 tokens) | Balanced recall and index size | May miss fine-grained facts | RAG for articles |
| Large (500+ tokens) | Preserve context | Coarse retrieval, fewer vectors | Summarization, long passages |

---

## Operational notes

- Backups: export collections regularly; store embeddings and metadata alongside source IDs to allow reconstruction.
- Monitoring: track query latency, memory footprint, and index size.
- Security: use access controls for production deployments; sanitize metadata and filter inputs to avoid injection risks.

---

## Glossary

- Embedding: dense numeric vector representing semantic content.
- ANN: approximate nearest neighbor — algorithms to quickly retrieve nearest vectors.
- RAG: retrieval-augmented generation.
- Collection: container for items in ChromaDB.

---

## Appendix: Extended diagrams and flows

### Ingestion pipeline (detailed)

```mermaid
flowchart TD
  subgraph Ingest
    A1[Raw Sources]
    A2[Fetcher / Scraper]
    A3[Preprocessor / Cleaner]
    A4[Chunker]
    A5[Embedder]
    A6[Chroma Client]
    A7[Collection (add/upsert)]
  end

  A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7
```

### Query + Rerank pipeline

```mermaid
flowchart LR
  Q[User Query] --> QE[Query Embedder]
  QE --> QDB[Chroma Query (top-k)]
  QDB --> RR[Reranker (optional)]
  RR --> LLM[LLM with Context]
  LLM --> Out[Answer]
```

---

## References and further reading

- Official ChromaDB docs and GitHub repository
- Papers on vector search & ANN algorithms (HNSW, IVF, Product Quantization)
- Tutorials on RAG and LLM retrieval patterns

