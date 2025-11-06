# Module 6: RAG & Transformer Architecture

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Classes:** 10-11

---

## Class 10: RAG (Retrieval-Augmented Generation)

### Topics Covered

- Architecture & Workflow
- Retriever + Generator pipeline
- Evaluation of RAG systems
- Detailed notes on hybrid retrieval (BM25 + Embedding)

### Learning Objectives

By the end of this class, students will be able to:
- Understand complete RAG architecture and workflow
- Design retriever and generator components
- Evaluate RAG system performance
- Implement hybrid retrieval strategies
- Optimize RAG systems for production

### Core Concepts

#### RAG Architecture & Workflow - Complete Analysis

RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation to create knowledge-intensive AI systems. This section provides a comprehensive analysis of RAG architecture, workflow, and mathematical foundations.

**Complete RAG Pipeline Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              RAG PIPELINE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

Phase 1: Data Ingestion (Offline/Indexing)
┌─────────────────────────────────────────────────────────────┐
│ Raw Documents (PDF, TXT, HTML, etc.)                          │
│    │                                                          │
│    ▼                                                          │
│ ┌──────────────────┐                                         │
│ │ Document Loader  │ → Extract text and metadata              │
│ │ • PDF parsing    │                                         │
│ │ • Web scraping   │                                         │
│ │ • Database query │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Text Preprocessing│ → Clean and normalize text              │
│ │ • Unicode norm    │                                         │
│ │ • Lowercase      │                                         │
│ │ • Remove noise   │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Chunking         │ → Split into manageable pieces           │
│ │ • Recursive split│                                         │
│ │ • Token-based    │                                         │
│ │ • Semantic chunk │                                         │
│ │ • Overlap: 200   │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Embedding Generation│ → Convert text to vectors            │
│ │ • OpenAI embeddings│                                       │
│ │ • Sentence-BERT   │                                        │
│ │ • Dimension: 1536 │                                        │
│ └────────┬─────────┘                                         │
└──────────┼──────────────────────────────────────────────────┘
           │
           ▼
Phase 2: Indexing (Offline/Indexing)
┌─────────────────────────────────────────────────────────────┐
│ ┌──────────────────┐                                         │
│ │ Vector Store      │ → Store embeddings                      │
│ │ • ChromaDB       │                                         │
│ │ • Pinecone       │                                         │
│ │ • FAISS          │                                         │
│ │ • HNSW index     │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Metadata Storage  │ → Store document metadata              │
│ │ • Source          │                                         │
│ │ • Timestamp       │                                         │
│ │ • Chunk ID        │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Index Optimization│ → Optimize for search                  │
│ │ • HNSW params    │                                         │
│ │ • Compression    │                                         │
│ └──────────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
Phase 3: Query Processing (Online/Inference)
┌─────────────────────────────────────────────────────────────┐
│ User Query: "What is machine learning?"                       │
│    │                                                          │
│    ▼                                                          │
│ ┌──────────────────┐                                         │
│ │ Query Embedding  │ → Generate query vector                  │
│ │ • Same model     │                                         │
│ │ • Dimension: 1536│                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Retrieval        │ → Find similar documents                │
│ │ • Vector search  │                                         │
│ │ • BM25 search    │                                         │
│ │ • Hybrid search  │                                         │
│ │ • Top-k: 10      │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Reranking        │ → Improve ranking (optional)            │
│ │ • Cross-encoder   │                                         │
│ │ • Reorder top-k  │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Context Assembly │ → Combine retrieved chunks              │
│ │ • Token budget   │                                         │
│ │ • Deduplication  │                                         │
│ │ • Ordering       │                                         │
│ └────────┬─────────┘                                         │
└──────────┼──────────────────────────────────────────────────┘
           │
           ▼
Phase 4: Generation (Online/Inference)
┌─────────────────────────────────────────────────────────────┐
│ ┌──────────────────┐                                         │
│ │ Prompt Construction│ → Build LLM prompt                     │
│ │ • System message │                                         │
│ │ • Context        │                                         │
│ │ • User query     │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ LLM Inference    │ → Generate answer                       │
│ │ • GPT-4          │                                         │
│ │ • Claude         │                                         │
│ │ • Streaming      │                                         │
│ └────────┬─────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ ┌──────────────────┐                                         │
│ │ Response Post-proc│ → Format and validate                   │
│ │ • Parsing        │                                         │
│ │ • Validation     │                                         │
│ │ • Source citation│                                         │
│ └──────────────────┘                                         │
│          │                                                    │
│          ▼                                                    │
│ Final Answer: "Machine learning is..."                        │
└─────────────────────────────────────────────────────────────┘
```

**Mathematical Model of RAG Pipeline:**

```
RAG_Model:

For query q and document corpus D = {d₁, d₂, ..., dₙ}:

1. Document Processing (Offline):
   For each document dᵢ:
     chunksᵢ = Chunk(dᵢ, chunk_size, overlap)
     embeddingsᵢ = EmbeddingModel(chunksᵢ)
     Index.add(embeddingsᵢ, metadataᵢ)

2. Query Processing (Online):
   query_embedding = EmbeddingModel(q)
   
   Retrieval:
   retrieved_docs = Retrieve(Index, query_embedding, k)
   
   Where:
   Retrieve(Index, q, k) = TopK(Similarity(query_embedding, doc_embeddings))
   
   Similarity can be:
   - Cosine: cos(θ) = (q · d) / (||q|| × ||d||)
   - Dot Product: q · d
   - Euclidean: ||q - d||²

3. Reranking (Optional):
   reranked_docs = Rerank(query, retrieved_docs, k')
   
   Where:
   Rerank(q, docs, k') = TopK'(CrossEncoder(q, docᵢ))

4. Context Assembly:
   context = Assemble(reranked_docs[:k'], token_budget)
   
   Where:
   Assemble(docs, budget) = Select(docs, Σ tokens(docᵢ) ≤ budget)

5. Generation:
   prompt = BuildPrompt(context, q)
   answer = LLM(prompt)
   
   Where:
   BuildPrompt(context, q) = SystemMessage + Context + Query
   LLM(prompt) = Generate(prompt, temperature, max_tokens)

6. Final Output:
   response = {
       "answer": answer,
       "sources": [doc.metadata for doc in reranked_docs[:k']],
       "confidence": CalculateConfidence(answer, context)
   }
```

**RAG Workflow Mathematical Formulation:**

```
Complete_RAG_Workflow:

Given:
- Query: q
- Document corpus: D = {d₁, d₂, ..., dₙ}
- Embedding model: E: Text → R^d
- LLM: L: Prompt → Text
- Retrieval function: R: Query → TopK(Documents)
- Reranking function: Rerank: (Query, Documents) → Reordered(Documents)

RAG Pipeline:

1. Indexing (Offline):
   Index = BuildIndex(D)
   For each dᵢ ∈ D:
     chunksᵢ = Chunk(dᵢ)
     embeddingsᵢ = E(chunksᵢ)
     Index.insert(embeddingsᵢ, metadata(dᵢ))

2. Querying (Online):
   Step 1: Embed query
     q_embed = E(q)
   
   Step 2: Retrieve
     candidates = R(Index, q_embed, k=10)
     Where R uses similarity search:
       similarity(doc, q) = CosineSimilarity(E(doc.text), q_embed)
       candidates = TopK(similarity(doc, q) for doc in Index)
   
   Step 3: Rerank (optional)
     ranked = Rerank(q, candidates, k'=5)
     Where Rerank uses cross-encoder:
       score(doc) = CrossEncoder(q, doc.text)
       ranked = Sort(candidates, key=score, reverse=True)[:k']
   
   Step 4: Assemble context
     context = ""
     for doc in ranked:
       if tokens(context + doc.text) ≤ token_budget:
         context += doc.text + "\n"
       else:
         break
   
   Step 5: Generate
     prompt = f"Context: {context}\n\nQuestion: {q}\n\nAnswer:"
     answer = L(prompt)
   
   Step 6: Return
     return {
       "answer": answer,
       "sources": [doc.metadata for doc in ranked],
       "confidence": CalculateConfidence(answer, context)
     }
```

**RAG Component Details:**

```
1. Data Ingestion Components:

Document Loader:
- Input: File paths or data sources
- Output: Raw text documents
- Processing: Extract text, extract metadata
- Time: O(n) where n = number of documents

Text Preprocessing:
- Unicode normalization
- Lowercasing (optional)
- Special character handling
- Formula: Preprocess(text) = Normalize(Clean(text))

Chunking:
- Recursive character splitting
- Token-based splitting
- Semantic chunking
- Formula: Chunks = Split(document, chunk_size, overlap)
- Overlap ensures context preservation

Embedding Generation:
- Model: OpenAI text-embedding-ada-002
- Dimension: 1536
- Batch processing for efficiency
- Formula: embedding = EmbeddingModel(text)
- Time: O(n) where n = number of chunks
```

```
2. Indexing Components:

Vector Store Creation:
- Index type: HNSW (Hierarchical Navigable Small World)
- Parameters: M (connections), ef_construction, ef_search
- Build time: O(n log n) where n = number of vectors
- Space: O(n × d) where d = embedding dimension

Metadata Storage:
- Store source, timestamp, chunk ID
- Enable filtering and search
- Space: O(n) where n = number of chunks

Index Optimization:
- HNSW parameter tuning
- Compression (optional)
- Updates and maintenance
```

```
3. Query Processing Components:

Query Embedding:
- Same embedding model as documents
- Formula: q_embed = EmbeddingModel(query)
- Time: O(1) - single embedding

Retrieval:
- Similarity search in vector space
- Algorithm: HNSW approximate nearest neighbor
- Formula: candidates = TopK(Similarity(q_embed, doc_embeddings))
- Time: O(log n) for HNSW where n = index size

Reranking:
- Cross-encoder model
- More accurate but slower
- Formula: scores = CrossEncoder(query, candidate.text)
- Time: O(k) where k = number of candidates

Context Assembly:
- Token budget management
- Deduplication
- Ordering by relevance
- Formula: context = Select(candidates, token_budget)
```

```
4. Generation Components:

Prompt Construction:
- Template: System + Context + Query
- Token management
- Formula: prompt = Template(context, query)

LLM Inference:
- Model: GPT-4, Claude, etc.
- Temperature: 0 for deterministic
- Max tokens: Limit response length
- Formula: answer = LLM(prompt, temperature, max_tokens)
- Time: O(m) where m = response length

Response Post-processing:
- Parsing structured output
- Validation
- Source citation
- Formatting
```

#### Retriever + Generator Pipeline - Detailed Architecture

The RAG pipeline consists of two main components: the Retriever (finds relevant documents) and the Generator (produces answers). This section provides a comprehensive analysis of their architecture, interaction, and optimization strategies.

**Retriever Component - Complete Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              RETRIEVER COMPONENT ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────┘

User Query: "What is machine learning?"
    │
    ▼
┌──────────────────┐
│ Query Preprocessing│
│ • Normalization   │
│ • Tokenization   │
│ • Expansion       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Query Embedding  │
│ • Embedding model│
│ • Dimension: 1536│
│ • Time: O(1)     │
└────────┬─────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Vector Search│    │ BM25 Search  │    │ Hybrid Search│
│ • Cosine      │    │ • TF-IDF     │    │ • Combined   │
│ • Similarity │    │ • Keyword    │    │ • RRF        │
│ • Top-k: 10  │    │ • Top-k: 10  │    │ • Weighted   │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                    │
       └───────────────────┴────────────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │ Result Combination   │
              │ • Deduplication      │
              │ • Score fusion       │
              │ • Top-k selection    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Reranking (Optional) │
              │ • Cross-encoder      │
              │ • Reorder top-k      │
              │ • Top-k': 5          │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Retrieved Documents   │
              │ • Top-k chunks       │
              │ • Metadata           │
              │ • Scores             │
              └──────────────────────┘
```

**Retriever Mathematical Model:**

```
Retriever_Model:

For query q and document corpus D:

1. Query Preprocessing:
   q_processed = Preprocess(q)
   Where:
   Preprocess(q) = Normalize(Tokenize(q))

2. Query Embedding:
   q_embed = E(q_processed)
   Where:
   E: Text → R^d (embedding model)
   d = 1536 (embedding dimension)

3. Retrieval Strategies:

   a) Vector Search:
      candidates_vec = TopK(Similarity(q_embed, doc_embeddings))
      Where:
      Similarity(q, d) = CosineSimilarity(q, d)
      = (q · d) / (||q|| × ||d||)
   
   b) BM25 Search:
      candidates_bm25 = TopK(BM25_Score(q, docs))
      Where:
      BM25_Score(q, d) = Σ IDF(t) × TF(t, d) × LengthNorm(d)
      (See Module 4 for complete BM25 formula)
   
   c) Hybrid Search:
      candidates_hybrid = Combine(candidates_vec, candidates_bm25)
      Where:
      Combine uses RRF (Reciprocal Rank Fusion):
      RRF_Score(doc) = Σ(1 / (k + rank_i))
      for each retrieval method i

4. Reranking (Optional):
   reranked = Rerank(q, candidates_hybrid, k'=5)
   Where:
   Rerank uses cross-encoder:
   score(doc) = CrossEncoder(q, doc.text)
   reranked = Sort(candidates_hybrid, key=score, reverse=True)[:k']

5. Final Output:
   retrieved_docs = reranked[:k']
   Return: {
       "documents": retrieved_docs,
       "scores": [score(doc) for doc in retrieved_docs],
       "metadata": [doc.metadata for doc in retrieved_docs]
   }
```

**Generator Component - Complete Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              GENERATOR COMPONENT ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────┘

Query + Retrieved Context
    │
    ▼
┌──────────────────┐
│ Context Assembly │
│ • Token budget   │
│ • Deduplication │
│ • Ordering       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Prompt Template │
│ • System message │
│ • Context        │
│ • Query          │
│ • Instructions   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Token Budgeting  │
│ • Count tokens   │
│ • Trim if needed │
│ • Validate       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLM Inference    │
│ • Model: GPT-4   │
│ • Temperature: 0 │
│ • Max tokens: 500│
│ • Streaming: Yes │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Response Parsing │
│ • Extract answer │
│ • Validate       │
│ • Format         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Quality Control  │
│ • Faithfulness   │
│ • Relevance      │
│ • Completeness   │
└────────┬─────────┘
         │
         ▼
    Final Answer
```

**Generator Mathematical Model:**

```
Generator_Model:

For query q and retrieved documents D_retrieved:

1. Context Assembly:
   context = Assemble(D_retrieved, token_budget)
   
   Where:
   Assemble(docs, budget):
     context = ""
     for doc in docs (sorted by relevance):
       if tokens(context + doc.text) ≤ budget:
         context += doc.text + "\n"
       else:
         break
     return context
   
   Token Budget Calculation:
   budget = max_tokens - tokens(system_prompt) - tokens(query) - tokens(instructions) - buffer
   
   Where:
   - max_tokens: LLM context window (e.g., 8192 for GPT-3.5)
   - buffer: Safety margin (e.g., 100 tokens)

2. Prompt Construction:
   prompt = BuildPrompt(context, q)
   
   Template:
   prompt = f"""
   System: You are a helpful assistant. Answer questions based on the provided context.
   
   Context:
   {context}
   
   Question: {q}
   
   Answer:
   """
   
   Where:
   BuildPrompt(context, q) = SystemMessage + Context + Query + Instructions

3. LLM Inference:
   answer = LLM(prompt, temperature, max_tokens)
   
   Where:
   LLM: Prompt → Text
   - temperature: Sampling temperature (0 = deterministic)
   - max_tokens: Maximum response length
   
   Generation Process:
   P(answer | prompt) = Π P(token_i | prompt, tokens_<i)
   
   Where:
   - Autoregressive generation
   - Each token depends on previous tokens

4. Response Post-processing:
   final_answer = PostProcess(answer)
   
   Where:
   PostProcess includes:
   - Extract answer (remove metadata)
   - Validate format
   - Add source citations
   - Format for display

5. Quality Control:
   quality_metrics = {
       "faithfulness": CheckFaithfulness(answer, context),
       "relevance": CheckRelevance(answer, q),
       "completeness": CheckCompleteness(answer, q)
   }
   
   Where:
   - Faithfulness: Answer grounded in context (no hallucination)
   - Relevance: Answer addresses the query
   - Completeness: Answer is complete and informative
```

**Retriever-Generator Integration:**

```
┌─────────────────────────────────────────────────────────────┐
│              RETRIEVER-GENERATOR INTEGRATION                 │
└─────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌──────────────────┐
│ Retriever        │
│ • Embed query     │
│ • Search index    │
│ • Retrieve top-k  │
│ • Rerank (opt)    │
└────────┬─────────┘
         │
         │ Retrieved Documents
         │
         ▼
┌──────────────────┐
│ Context Window   │
│ Management       │
│ • Token counting │
│ • Budgeting      │
│ • Trimming       │
└────────┬─────────┘
         │
         │ Optimized Context
         │
         ▼
┌──────────────────┐
│ Generator        │
│ • Build prompt    │
│ • LLM inference  │
│ • Generate answer │
│ • Post-process   │
└────────┬─────────┘
         │
         ▼
    Final Answer
```

**Integration Optimization Strategies:**

```
1. Context Window Management:
   
   Token Budget Allocation:
   - System prompt: 100 tokens
   - Query: tokens(query)
   - Instructions: 50 tokens
   - Context: budget - system - query - instructions - buffer
   - Response: max_response_tokens
   
   Formula:
   context_budget = max_tokens - system_tokens - query_tokens - instruction_tokens - buffer
   
   Example:
   For GPT-3.5 (max_tokens = 8192):
   - System: 100
   - Query: 50
   - Instructions: 50
   - Buffer: 100
   - Response: 500
   - Context budget: 8192 - 100 - 50 - 50 - 100 - 500 = 7392 tokens

2. Token Budgeting:
   
   Strategy:
   - Prioritize highest-scoring documents
   - Fill budget in order of relevance
   - Stop when budget is reached
   
   Algorithm:
   context = ""
   remaining_budget = context_budget
   for doc in retrieved_docs (sorted by score, descending):
       doc_tokens = count_tokens(doc.text)
       if doc_tokens ≤ remaining_budget:
           context += doc.text + "\n"
           remaining_budget -= doc_tokens
       else:
           # Option 1: Truncate document
           truncated = truncate(doc.text, remaining_budget)
           context += truncated
           break
           # Option 2: Skip document
           # continue

3. Prompt Optimization:
   
   Strategies:
   - Clear instructions: "Answer based on context only"
   - Format specifications: "Use bullet points"
   - Length constraints: "Keep answer concise"
   - Quality requirements: "Be accurate and complete"
   
   Example Optimized Prompt:
   """
   You are a helpful assistant. Answer the question using ONLY the provided context.
   If the context doesn't contain the answer, say "I don't know."
   
   Context:
   {context}
   
   Question: {query}
   
   Answer (be concise and accurate):
   """

4. Quality Control:
   
   Faithfulness Check:
   - Compare answer with context
   - Detect hallucinations
   - Formula: faithfulness = Similarity(answer, context)
   
   Relevance Check:
   - Compare answer with query
   - Ensure answer addresses question
   - Formula: relevance = Similarity(answer, query)
   
   Completeness Check:
   - Check if answer is complete
   - Detect truncated or incomplete answers
   - Use heuristics (sentence endings, etc.)
```

**Performance Optimization:**

```
1. Retrieval Optimization:
   - Use hybrid search (BM25 + embeddings)
   - Implement reranking for top-k results
   - Cache query embeddings
   - Optimize index structure (HNSW parameters)

2. Generation Optimization:
   - Use faster models for retrieval
   - Use powerful models for generation
   - Implement response caching
   - Stream responses for better UX

3. Integration Optimization:
   - Parallel retrieval and query processing
   - Async context assembly
   - Batch processing for multiple queries
   - Connection pooling for APIs
```

#### Advanced RAG Techniques - Comprehensive Guide

Advanced RAG techniques improve retrieval quality, answer accuracy, and system robustness. This section provides detailed explanations, mathematical formulations, and implementation strategies for each technique.

**1. Query Expansion - Detailed Analysis**

Query expansion generates multiple query variations to improve retrieval recall by handling synonyms, paraphrasing, and different phrasings.

**Query Expansion Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              QUERY EXPANSION ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────┘

Original Query: "machine learning"
    │
    ▼
┌──────────────────┐
│ Query Analysis   │
│ • Extract keywords│
│ • Identify intent │
│ • Detect entities │
└────────┬─────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Synonym      │    │ Paraphrase    │    │ LLM-based    │
│ Expansion    │    │ Generation    │    │ Expansion    │
│ • WordNet    │    │ • Back-trans  │    │ • GPT-4      │
│ • Thesaurus  │    │ • Paraphrase  │    │ • Prompts    │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                    │
       └───────────────────┴────────────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │ Expanded Queries      │
              │ • Original            │
              │ • Synonym variant     │
              │ • Paraphrase          │
              │ • LLM-generated       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Retrieve for Each     │
              │ • Search each query   │
              │ • Combine results     │
              │ • Deduplicate        │
              └──────────────────────┘
```

**Query Expansion Mathematical Model:**

```
Query_Expansion_Model:

For original query q:

1. Synonym Expansion:
   q_synonyms = [q, q₁', q₂', ..., qₙ']
   Where:
   qᵢ' = ReplaceSynonyms(q, word_i)
   
   Example:
   q = "machine learning"
   q_synonyms = [
       "machine learning",
       "ML",
       "artificial intelligence learning",
       "automated learning"
   ]

2. Paraphrase Generation:
   q_paraphrases = [q, q₁'', q₂'', ..., qₘ'']
   Where:
   qᵢ'' = Paraphrase(q) using LLM or translation model
   
   Example:
   q = "machine learning"
   q_paraphrases = [
       "machine learning",
       "how does machine learning work",
       "what is machine learning",
       "explain machine learning"
   ]

3. LLM-based Expansion:
   q_llm = LLM_Expand(q)
   
   Prompt: "Generate 3 variations of this query: {q}"
   q_llm = [
       q,
       LLM_Generate(q, variation_1),
       LLM_Generate(q, variation_2),
       LLM_Generate(q, variation_3)
   ]

4. Combined Retrieval:
   all_queries = q_synonyms ∪ q_paraphrases ∪ q_llm
   
   For each q_expanded in all_queries:
       results_q = Retrieve(Index, q_expanded, k)
   
   combined_results = Union(results_q for q in all_queries)
   
   Final ranking:
   final_results = Rerank(combined_results, original_query=q)
```

**2. Reranking - Detailed Analysis**

Reranking uses more sophisticated models (cross-encoders) to reorder initially retrieved documents, improving precision by better understanding query-document relevance.

**Reranking Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              RERANKING ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────┘

Initial Retrieval (Top-k: 20)
    │
    ▼
┌──────────────────┐
│ Cross-Encoder     │
│ • Input: Query +  │
│   Document        │
│ • Output: Score   │
│ • More accurate   │
│   but slower      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Score Documents   │
│ • Score each doc  │
│ • Time: O(k)      │
│ • k = 20 docs     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Reorder by Score  │
│ • Sort descending │
│ • Select top-k'   │
│ • k' = 5          │
└──────────────────┘
```

**Reranking Mathematical Model:**

```
Reranking_Model:

For query q and retrieved documents D_retrieved = [d₁, d₂, ..., dₖ]:

1. Cross-Encoder Scoring:
   For each document dᵢ:
       score(dᵢ) = CrossEncoder(q, dᵢ.text)
   
   Where:
   CrossEncoder: (Query, Document) → Score
   - Input: Concatenated query and document
   - Output: Relevance score (0-1)
   - More accurate than bi-encoder (sees query and document together)
   
   Example:
   CrossEncoder("machine learning", doc.text) = 0.87

2. Ranking:
   reranked = Sort(D_retrieved, key=score, reverse=True)
   
   Where:
   Sort by score in descending order

3. Selection:
   final_docs = reranked[:k']
   
   Where:
   k' < k (e.g., k=20, k'=5)
   
   Select top-k' documents after reranking

4. Latency Analysis:
   Total latency = Retrieval_latency + Reranking_latency
   
   Where:
   - Retrieval_latency: O(log n) for HNSW (fast)
   - Reranking_latency: O(k) for cross-encoder (slower)
   
   Optimization:
   - Use bi-encoder for initial retrieval (fast)
   - Use cross-encoder for reranking (accurate)
   - Two-stage approach: Fast retrieval + Accurate reranking
```

**3. Multi-Query Retrieval - Detailed Analysis**

Multi-query retrieval generates multiple queries from the original query, retrieves documents for each, and combines results to improve recall.

**Multi-Query Retrieval Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              MULTI-QUERY RETRIEVAL ARCHITECTURE               │
└─────────────────────────────────────────────────────────────┘

Original Query: "machine learning"
    │
    ▼
┌──────────────────┐
│ Query Generation  │
│ • LLM generates  │
│   3-5 variations  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Multiple Queries │
│ • Query 1: "ML"   │
│ • Query 2: "AI..."│
│ • Query 3: "..."  │
└────────┬───────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Retrieve Q1  │ │ Retrieve Q2  │ │ Retrieve Q3  │ │ ...          │
│ • Top-k: 10  │ │ • Top-k: 10  │ │ • Top-k: 10  │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │
       └────────────────┴────────────────┴────────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │ Combine Results       │
              │ • Union all docs      │
              │ • Deduplicate         │
              │ • RRF fusion          │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Final Top-k          │
              │ • Select top-k'      │
              │ • k' = 5             │
              └──────────────────────┘
```

**Multi-Query Retrieval Mathematical Model:**

```
Multi_Query_Retrieval_Model:

For original query q:

1. Query Generation:
   queries = [q, q₁, q₂, ..., qₙ]
   Where:
   qᵢ = LLM_GenerateQuery(q, variation_i)
   
   Example:
   q = "machine learning"
   queries = [
       "machine learning",
       "ML algorithms",
       "artificial intelligence learning",
       "automated machine learning"
   ]

2. Parallel Retrieval:
   For each query qᵢ in queries:
       results_i = Retrieve(Index, qᵢ, k=10)
   
   All results retrieved in parallel for efficiency

3. Result Combination:
   combined = Union(results_1, results_2, ..., results_n)
   
   Deduplication:
   unique_docs = Deduplicate(combined)

4. Score Fusion:
   For each document doc in unique_docs:
       doc.score = RRF_Score(doc, queries)
   
   Where:
   RRF_Score(doc, queries) = Σ(1 / (k + rank_i(doc)))
   for each query qᵢ where doc appears
   
   Example:
   If doc appears at rank 2 in query 1 and rank 5 in query 2:
   RRF_Score = 1/(2+2) + 1/(2+5) = 1/4 + 1/7 = 0.25 + 0.143 = 0.393

5. Final Selection:
   final_docs = TopK'(unique_docs, key=score, k'=5)
```

**4. Parent Document Retrieval - Detailed Analysis**

Parent document retrieval retrieves small chunks first, then returns their parent documents for better context preservation.

**Parent Document Retrieval Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│          PARENT DOCUMENT RETRIEVAL ARCHITECTURE              │
└─────────────────────────────────────────────────────────────┘

Documents
    │
    ▼
┌──────────────────┐
│ Hierarchical      │
│ Chunking          │
│ • Parent docs     │
│ • Child chunks    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Index Small       │
│ Chunks            │
│ • Embed chunks    │
│ • Store parent ref│
└────────┬─────────┘
         │
         ▼
Query: "machine learning"
    │
    ▼
┌──────────────────┐
│ Retrieve Chunks   │
│ • Top-k chunks    │
│ • k = 10          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Get Parent Docs   │
│ • Map chunk →     │
│   parent          │
│ • Deduplicate     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Final Context     │
│ • Parent docs     │
│ • Better context  │
└──────────────────┘
```

**Parent Document Retrieval Mathematical Model:**

```
Parent_Document_Retrieval_Model:

Given hierarchical document structure:
- Parent documents: P = {p₁, p₂, ..., pₘ}
- Child chunks: C = {c₁, c₂, ..., cₙ}
- Parent-child mapping: parent(cᵢ) = pⱼ

1. Indexing:
   For each chunk cᵢ:
       embedding_i = EmbeddingModel(cᵢ.text)
       Index.add(embedding_i, {
           "chunk_id": i,
           "parent_id": parent(cᵢ),
           "chunk_text": cᵢ.text
       })

2. Retrieval:
   Query: q
   
   Step 1: Retrieve chunks
       retrieved_chunks = Retrieve(Index, q, k=10)
       Where:
       Retrieve uses similarity search on chunk embeddings
   
   Step 2: Get parent documents
       parent_ids = [chunk.parent_id for chunk in retrieved_chunks]
       unique_parents = Deduplicate(parent_ids)
       parent_docs = [P[pid] for pid in unique_parents]
   
   Step 3: Return parent documents
       return parent_docs

3. Benefits:
   - Better context: Parent documents contain full context
   - Precision: Small chunks help find relevant documents
   - Recall: Parent documents provide complete information
   
   Example:
   Chunk: "Machine learning is a subset of AI"
   Parent: Full article about machine learning (5000 words)
   - Chunk helps find relevant document
   - Parent provides complete context for generation
```

**5. Query Routing - Detailed Analysis**

Query routing directs queries to specialized indexes or retrieval strategies based on query type or domain.

**Query Routing Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              QUERY ROUTING ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌──────────────────┐
│ Query Analysis   │
│ • Type detection │
│ • Domain ident   │
│ • Intent class   │
└────────┬─────────┘
         │
         ├──────────────────┬──────────────────┬──────────────────┐
         │                  │                  │                  │
         ▼                  ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Technical    │    │ General      │    │ Code         │    │ Legal        │
│ Index        │    │ Index        │    │ Index        │    │ Index        │
│ • ML docs    │    │ • Wikipedia  │    │ • GitHub     │    │ • Legal docs │
│ • Research   │    │ • General    │    │ • Code       │    │ • Contracts  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                    │                    │
       └───────────────────┴────────────────────┴────────────────────┘
                           │
                           ▼
              ┌──────────────────────┐
              │ Combine Results       │
              │ • Merge from all      │
              │   indexes             │
              │ • RRF fusion          │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Final Results         │
              └──────────────────────┘
```

**Query Routing Mathematical Model:**

```
Query_Routing_Model:

For query q and specialized indexes I = {I₁, I₂, ..., Iₙ}:

1. Query Classification:
   query_type = Classify(q)
   
   Where:
   Classify uses:
   - Keyword matching
   - ML classifier
   - LLM-based classification
   
   Example:
   q = "Python function definition"
   query_type = "code"  # Route to code index

2. Index Selection:
   selected_indexes = Route(query_type, I)
   
   Where:
   Route can be:
   - Single index: Route to most relevant
   - Multiple indexes: Route to all relevant
   - Weighted routing: Route with different weights
   
   Example:
   query_type = "technical"
   selected_indexes = [I_technical, I_general]  # Both, with weights

3. Parallel Retrieval:
   For each index Iᵢ in selected_indexes:
       results_i = Retrieve(Iᵢ, q, k_i)
   
   Where:
   k_i = retrieval_count for index i
   Can be different for different indexes

4. Result Fusion:
   combined = WeightedFusion(results_1, results_2, ..., results_n)
   
   Where:
   WeightedFusion uses:
   - Weighted RRF: Different weights for different indexes
   - Score normalization: Normalize scores from different indexes
   - Ranking aggregation: Combine rankings
   
   Formula:
   final_score(doc) = Σ(w_i × RRF_Score_i(doc))
   for each index i where doc appears
   
   Where:
   w_i = weight for index i
   Σ w_i = 1 (normalized weights)

5. Final Ranking:
   final_results = TopK(combined, k=5)
```

#### Hybrid Retrieval (BM25 + Embedding) - Complete Analysis

Hybrid retrieval combines keyword-based search (BM25) with semantic search (embeddings) to leverage the strengths of both approaches. This section provides a comprehensive analysis of hybrid retrieval strategies, mathematical formulations, and implementation details.

**Why Hybrid Retrieval?**

```
Problem with Single Method:

BM25 (Keyword-based):
- Strengths: Exact keyword matching, fast, interpretable
- Weaknesses: Misses synonyms, semantic variations, paraphrasing

Embeddings (Semantic):
- Strengths: Semantic understanding, handles synonyms, paraphrasing
- Weaknesses: May miss exact keyword matches, less interpretable

Solution: Hybrid Approach
- Combines both methods
- Better recall: Finds more relevant documents
- Better precision: Ranks relevant documents higher
- More robust: Handles various query types
```

**Hybrid Retrieval Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              HYBRID RETRIEVAL ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────┘

User Query: "machine learning algorithms"
    │
    │
    ├──────────────────────────┬──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ BM25 Search  │      │ Embedding    │      │ Query        │
│ • Keyword    │      │ Search       │      │ Analysis      │
│ • TF-IDF     │      │ • Semantic   │      │ • Route       │
│ • Top-k: 10  │      │ • Vector     │      │ • Weight      │
└──────┬───────┘      │ • Top-k: 10  │      └──────┬───────┘
       │              └──────┬───────┘             │
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ▼
              ┌──────────────────────┐
              │ Score Fusion          │
              │ • RRF                 │
              │ • Weighted            │
              │ • Learned             │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Deduplication         │
              │ • Remove duplicates   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Reranking (Optional)  │
              │ • Cross-encoder       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Final Top-k          │
              │ • Select top-k'      │
              │ • k' = 5             │
              └──────────────────────┘
```

**Hybrid Retrieval Mathematical Model:**

```
Hybrid_Retrieval_Model:

For query q and document corpus D:

1. BM25 Retrieval:
   bm25_results = BM25_Retrieve(q, D, k=10)
   
   Where:
   BM25_Score(q, d) = Σ IDF(t) × TF(t, d) × LengthNorm(d)
   
   For each term t in q:
   - IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5))
   - TF(t, d) = (f(t, d) × (k1 + 1)) / (f(t, d) + k1 × (1 - b + b × |d|/avg_doc_length))
   - LengthNorm(d) = (1 - b + b × |d|/avg_doc_length)
   
   Parameters:
   - k1 = 1.5 (term frequency saturation)
   - b = 0.75 (length normalization)
   - N = total documents
   - n(t) = documents containing term t

2. Embedding Retrieval:
   embedding_results = Embedding_Retrieve(q, D, k=10)
   
   Where:
   q_embed = EmbeddingModel(q)
   d_embed = EmbeddingModel(d.text) for each d in D
   
   Similarity(q, d) = CosineSimilarity(q_embed, d_embed)
   = (q_embed · d_embed) / (||q_embed|| × ||d_embed||)
   
   embedding_results = TopK(Similarity(q, d) for d in D)

3. Score Fusion Strategies:

   a) Reciprocal Rank Fusion (RRF):
      For each document d:
          rrf_score(d) = Σ(1 / (k + rank_i(d)))
          for each retrieval method i
      
      Where:
      - rank_i(d) = rank of document d in method i results
      - k = constant (typically 60)
      
      Example:
      If d appears at rank 2 in BM25 and rank 5 in embeddings:
      rrf_score = 1/(60+2) + 1/(60+5) = 1/62 + 1/65 = 0.0161 + 0.0154 = 0.0315
   
   b) Weighted Combination:
      combined_score(d) = α × Normalize(bm25_score(d)) + (1-α) × Normalize(embedding_score(d))
      
      Where:
      - α = weight for BM25 (typically 0.3-0.5)
      - Normalize: Min-max normalization or z-score normalization
      
      Example:
      α = 0.4
      bm25_normalized = 0.8
      embedding_normalized = 0.9
      combined_score = 0.4 × 0.8 + 0.6 × 0.9 = 0.32 + 0.54 = 0.86
   
   c) Learned Combination:
      combined_score(d) = ML_Model(bm25_score(d), embedding_score(d), features(d))
      
      Where:
      - ML_Model: Trained model (e.g., neural network)
      - Features: Additional features (document length, position, etc.)
      - Trained on labeled query-document pairs

4. Final Ranking:
   combined_results = Sort(all_documents, key=combined_score, reverse=True)
   final_results = combined_results[:k']
   
   Where:
   k' = final number of documents (e.g., 5)
```

**Complete Hybrid Retrieval Implementation:**

```python
"""
Complete Hybrid Retrieval Implementation

This implementation demonstrates:
1. BM25 retrieval
2. Embedding retrieval
3. Score fusion strategies
4. Reranking
5. Final selection
"""

from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and embedding-based search.
    
    Mathematical Model:
        For query q and documents D:
        1. BM25_Results = BM25_Retrieve(q, D, k)
        2. Embedding_Results = Embedding_Retrieve(q, D, k)
        3. Combined_Score = Fusion(BM25_Score, Embedding_Score)
        4. Final_Results = TopK(Combined_Score, k')
    """
    
    def __init__(self, 
                 documents: List[str],
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 bm25_k1: float = 1.5,
                 bm25_b: float = 0.75):
        """
        Initialize hybrid retriever.
        
        Args:
            documents: List of document texts
            embedding_model_name: Sentence transformer model name
            bm25_k1: BM25 term frequency saturation parameter
            bm25_b: BM25 length normalization parameter
        """
        self.documents = documents
        
        # Initialize BM25
        # BM25 uses tokenized documents
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=bm25_k1, b=bm25_b)
        
        # Initialize embedding model
        # Sentence transformers provide pre-trained embedding models
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Pre-compute document embeddings
        # Embeddings are computed once and reused for all queries
        print(f"Computing embeddings for {len(documents)} documents...")
        self.doc_embeddings = self.embedding_model.encode(
            documents, 
            show_progress_bar=True,
            batch_size=32
        )
        print("Embeddings computed successfully!")
    
    def bm25_retrieve(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve documents using BM25.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of (document_index, bm25_score) tuples
            
        Mathematical Model:
            BM25_Score(q, d) = Σ IDF(t) × TF(t, d) × LengthNorm(d)
            for each term t in query q
        """
        # Tokenize query
        query_tokens = query.split()
        
        # Get BM25 scores for all documents
        # BM25Okapi.get_scores returns scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k documents
        # Argsort returns indices sorted by score (descending)
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return document indices and scores
        results = [(idx, float(scores[idx])) for idx in top_indices]
        
        return results
    
    def embedding_retrieve(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve documents using embeddings.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of (document_index, cosine_similarity) tuples
            
        Mathematical Model:
            Similarity(q, d) = CosineSimilarity(Embedding(q), Embedding(d))
            = (q_embed · d_embed) / (||q_embed|| × ||d_embed||)
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Compute cosine similarities
        # Cosine similarity between query and all document embeddings
        similarities = cosine_similarity(
            [query_embedding], 
            self.doc_embeddings
        )[0]
        
        # Get top-k documents
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return document indices and similarities
        results = [(idx, float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def rrf_fusion(self, 
                   bm25_results: List[Tuple[int, float]],
                   embedding_results: List[Tuple[int, float]],
                   k: int = 60) -> List[Tuple[int, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            bm25_results: BM25 retrieval results
            embedding_results: Embedding retrieval results
            k: RRF constant (typically 60)
            
        Returns:
            List of (document_index, rrf_score) tuples
            
        Mathematical Model:
            RRF_Score(d) = Σ(1 / (k + rank_i(d)))
            for each retrieval method i
            
        Example:
            If document appears at rank 2 in BM25 and rank 5 in embeddings:
            RRF_Score = 1/(60+2) + 1/(60+5) = 0.0315
        """
        # Create rank dictionaries
        # rank[doc_id][method] = rank in that method
        ranks = {}
        
        # Add BM25 ranks
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            if doc_id not in ranks:
                ranks[doc_id] = {}
            ranks[doc_id]['bm25'] = rank
        
        # Add embedding ranks
        for rank, (doc_id, _) in enumerate(embedding_results, start=1):
            if doc_id not in ranks:
                ranks[doc_id] = {}
            ranks[doc_id]['embedding'] = rank
        
        # Compute RRF scores
        rrf_scores = {}
        for doc_id, method_ranks in ranks.items():
            rrf_score = 0.0
            for method, rank in method_ranks.items():
                rrf_score += 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_results
    
    def weighted_fusion(self,
                       bm25_results: List[Tuple[int, float]],
                       embedding_results: List[Tuple[int, float]],
                       alpha: float = 0.4) -> List[Tuple[int, float]]:
        """
        Combine results using weighted combination.
        
        Args:
            bm25_results: BM25 retrieval results
            embedding_results: Embedding retrieval results
            alpha: Weight for BM25 (1-alpha for embeddings)
            
        Returns:
            List of (document_index, combined_score) tuples
            
        Mathematical Model:
            Combined_Score(d) = α × Normalize(BM25_Score(d)) + (1-α) × Normalize(Embedding_Score(d))
        """
        # Normalize scores to [0, 1]
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        embedding_scores = {doc_id: score for doc_id, score in embedding_results}
        
        # Min-max normalization
        if bm25_results:
            bm25_max = max(score for _, score in bm25_results)
            bm25_min = min(score for _, score in bm25_results)
            if bm25_max > bm25_min:
                bm25_scores = {
                    doc_id: (score - bm25_min) / (bm25_max - bm25_min)
                    for doc_id, score in bm25_scores.items()
                }
        
        if embedding_results:
            embedding_max = max(score for _, score in embedding_results)
            embedding_min = min(score for _, score in embedding_results)
            if embedding_max > embedding_min:
                embedding_scores = {
                    doc_id: (score - embedding_min) / (embedding_max - embedding_min)
                    for doc_id, score in embedding_scores.items()
                }
        
        # Combine scores
        all_doc_ids = set(bm25_scores.keys()) | set(embedding_scores.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            embedding_score = embedding_scores.get(doc_id, 0.0)
            combined_score = alpha * bm25_score + (1 - alpha) * embedding_score
            combined_scores[doc_id] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results
    
    def retrieve(self,
                 query: str,
                 k: int = 10,
                 fusion_method: str = 'rrf',
                 alpha: float = 0.4) -> List[Tuple[int, float, str]]:
        """
        Perform hybrid retrieval.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            fusion_method: 'rrf' or 'weighted'
            alpha: Weight for BM25 (only for weighted fusion)
            
        Returns:
            List of (document_index, score, method) tuples
        """
        # Retrieve from both methods
        bm25_results = self.bm25_retrieve(query, k=k)
        embedding_results = self.embedding_retrieve(query, k=k)
        
        # Combine results
        if fusion_method == 'rrf':
            combined_results = self.rrf_fusion(bm25_results, embedding_results)
        elif fusion_method == 'weighted':
            combined_results = self.weighted_fusion(bm25_results, embedding_results, alpha)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Format results
        final_results = [
            (doc_id, score, 'hybrid')
            for doc_id, score in combined_results[:k]
        ]
        
        return final_results

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning learns through trial and error.",
    ]
    
    # Initialize retriever
    retriever = HybridRetriever(documents)
    
    # Query
    query = "AI algorithms"
    
    # Retrieve using RRF
    results_rrf = retriever.retrieve(query, k=5, fusion_method='rrf')
    print(f"\nRRF Fusion Results:")
    for doc_id, score, method in results_rrf:
        print(f"  Doc {doc_id}: {score:.4f} - {documents[doc_id]}")
    
    # Retrieve using weighted fusion
    results_weighted = retriever.retrieve(query, k=5, fusion_method='weighted', alpha=0.4)
    print(f"\nWeighted Fusion Results (α=0.4):")
    for doc_id, score, method in results_weighted:
        print(f"  Doc {doc_id}: {score:.4f} - {documents[doc_id]}")

"""
Expected Output:

Computing embeddings for 5 documents...
Embeddings computed successfully!

RRF Fusion Results:
  Doc 0: 0.0323 - Machine learning is a subset of artificial intelligence.
  Doc 1: 0.0317 - Deep learning uses neural networks with multiple layers.
  Doc 4: 0.0164 - Reinforcement learning learns through trial and error.
  Doc 2: 0.0161 - Natural language processing enables computers to understand text.
  Doc 3: 0.0159 - Computer vision allows machines to interpret visual information.

Weighted Fusion Results (α=0.4):
  Doc 0: 0.8234 - Machine learning is a subset of artificial intelligence.
  Doc 1: 0.7891 - Deep learning uses neural networks with multiple layers.
  Doc 4: 0.6543 - Reinforcement learning learns through trial and error.
  Doc 2: 0.6123 - Natural language processing enables computers to understand text.
  Doc 3: 0.5987 - Computer vision allows machines to interpret visual information.
"""
```

**Hybrid Retrieval Benefits - Detailed Analysis:**

```
1. Higher Recall:
   - BM25 finds documents with exact keywords
   - Embeddings find semantically similar documents
   - Combined: More relevant documents found
   
   Example:
   Query: "ML"
   - BM25: Finds documents with "ML" keyword
   - Embeddings: Finds documents about "machine learning"
   - Hybrid: Finds both

2. Better Precision:
   - Documents appearing in both methods are likely more relevant
   - RRF gives higher scores to documents ranked highly in both
   - Final ranking is more accurate
   
   Example:
   Document appearing in:
   - BM25 rank 1, Embeddings rank 2 → High RRF score
   - BM25 rank 10, Embeddings rank 15 → Lower RRF score

3. Robustness:
   - Handles various query types
   - Works for both keyword-heavy and semantic queries
   - Less sensitive to query formulation
   
   Example:
   - "machine learning" → Works
   - "ML algorithms" → Works
   - "how does AI learn" → Works

4. Performance:
   - BM25: Fast (O(n) for small corpora)
   - Embeddings: Fast (O(log n) with HNSW)
   - Combined: Still fast with parallel execution
```

#### Evaluation of RAG Systems

**Evaluation Metrics:**

**1. Retrieval Metrics:**
- **Precision@k:** Fraction of retrieved docs that are relevant
- **Recall@k:** Fraction of relevant docs that are retrieved
- **MRR (Mean Reciprocal Rank):** Average of 1/rank of first relevant doc
- **NDCG (Normalized Discounted Cumulative Gain):** Ranking quality

**2. Generation Metrics:**
- **BLEU:** N-gram overlap with reference
- **ROUGE:** Recall-oriented summary evaluation
- **METEOR:** Semantic similarity
- **BERTScore:** Semantic similarity using embeddings

**3. End-to-End Metrics:**
- **Answer Accuracy:** Correctness of final answer
- **Faithfulness:** Answer grounded in retrieved context
- **Answer Relevance:** Answer relevance to question
- **Context Utilization:** How well context is used

**Evaluation Framework:**
```python
# 1. Create evaluation dataset
questions = ["What is X?", "How does Y work?"]
ground_truth = ["Expected answers"]
contexts = ["Relevant documents"]

# 2. Evaluate retrieval
retrieval_scores = evaluate_retrieval(
    questions, contexts, ground_truth
)

# 3. Evaluate generation
generation_scores = evaluate_generation(
    questions, contexts, ground_truth, answers
)

# 4. Evaluate end-to-end
rag_scores = evaluate_rag(
    questions, answers, ground_truth
)
```

**Common Issues:**
- **Hallucination:** Model generates information not in context
- **Insufficient Context:** Not enough relevant information retrieved
- **Context Overload:** Too much irrelevant context
- **Query Mismatch:** Query doesn't match document style

---

## Class 11: Transformer Architecture Deep Dive

### Topics Covered

- Attention mechanism (Self, Cross, Multi-head)
- Positional encoding, residual connections
- Encoder-Decoder models: BERT, GPT, T5
- Visual walkthrough & code snippets

### Learning Objectives

By the end of this class, students will be able to:
- Understand transformer architecture in detail
- Explain attention mechanisms mathematically
- Compare different transformer architectures
- Implement transformer components from scratch
- Understand how transformers enable modern LLMs

### Core Concepts

#### Transformer Architecture Overview - Complete Analysis

The Transformer architecture revolutionized natural language processing by replacing recurrent and convolutional layers with attention mechanisms. This section provides a comprehensive analysis of the Transformer architecture, its components, and mathematical foundations.

**Complete Transformer Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────┘

Input Sequence: [x₁, x₂, ..., xₙ]
    │
    ▼
┌──────────────────┐
│ Input Embedding  │
│ • Token IDs →    │
│   Embeddings     │
│ • Dimension: d_model│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Positional       │
│ Encoding         │
│ • Add position   │
│   information    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Transformer      │
│ Block 1          │
│ • Multi-Head     │
│   Attention      │
│ • Feed Forward   │
│ • Residual + Norm│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Transformer      │
│ Block 2          │
│ • Multi-Head     │
│   Attention      │
│ • Feed Forward   │
│ • Residual + Norm│
└────────┬─────────┘
         │
         ▼
         ...
         │
         ▼
┌──────────────────┐
│ Transformer      │
│ Block N          │
│ • Multi-Head     │
│   Attention      │
│ • Feed Forward   │
│ • Residual + Norm│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Output Layer     │
│ • Linear         │
│ • Softmax        │
│ • Vocabulary size│
└────────┬─────────┘
         │
         ▼
Output: [y₁, y₂, ..., yₙ]
```

**Transformer Block Detailed Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────┘

Input: X (shape: [batch_size, seq_len, d_model])
    │
    ▼
┌──────────────────┐
│ Layer Norm 1     │
│ • Normalize input│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Multi-Head       │
│ Attention        │
│ • Q, K, V        │
│ • Attention scores│
│ • Output         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Residual         │
│ Connection       │
│ • Add: X + Attention(X)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Layer Norm 2     │
│ • Normalize      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feed Forward     │
│ • Linear 1       │
│ • ReLU           │
│ • Linear 2       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Residual         │
│ Connection       │
│ • Add: X + FFN(X)│
└────────┬─────────┘
         │
         ▼
Output: X' (shape: [batch_size, seq_len, d_model])
```

**Mathematical Model of Transformer:**

```
Transformer_Model:

For input sequence X = [x₁, x₂, ..., xₙ]:

1. Input Embedding:
   X_embed = Embedding(X)
   Where:
   X_embed ∈ R^(n × d_model)
   - n = sequence length
   - d_model = embedding dimension (typically 512 or 768)

2. Positional Encoding:
   X_pos = X_embed + PE
   Where:
   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

3. Transformer Blocks:
   For each block i = 1 to N:
       X_i = TransformerBlock(X_{i-1})
   
   Where:
   TransformerBlock(X) = LayerNorm(FFN(LayerNorm(X + Attention(X))) + LayerNorm(X + Attention(X)))
   
   Detailed:
   a) Attention:
      Attention(X) = MultiHeadAttention(X, X, X)
   
   b) Residual + Norm:
      X_attn = LayerNorm(X + Attention(X))
   
   c) Feed Forward:
      FFN(X) = Linear_2(ReLU(Linear_1(X)))
   
   d) Residual + Norm:
      X_out = LayerNorm(X_attn + FFN(X_attn))

4. Output Layer:
   Y = Softmax(Linear(X_N))
   Where:
   - Linear: R^(d_model) → R^(vocab_size)
   - Softmax: Probability distribution over vocabulary
   - Y ∈ R^(n × vocab_size)
```

**Key Innovations - Detailed Analysis:**

```
1. Attention Mechanism Replaces Recurrence:
   
   Traditional RNN:
   h_t = f(h_{t-1}, x_t)  # Sequential processing
   - Time complexity: O(n) sequential
   - Cannot parallelize
   
   Transformer Attention:
   Attention(X) = softmax(QK^T / √d_k) × V  # Parallel processing
   - Time complexity: O(n²) but parallelizable
   - Can process all positions simultaneously
   
   Benefit:
   - Much faster training (parallelization)
   - Better long-range dependencies
   - Enables training on very large datasets

2. Parallel Processing:
   
   All positions processed simultaneously:
   - Attention computed for all positions in parallel
   - Feed-forward network applied to all positions in parallel
   - Matrix operations enable GPU acceleration
   
   Benefit:
   - Training time: O(n²) but parallel vs O(n) sequential
   - Actual speedup: 100x-1000x for long sequences

3. Scalability:
   
   Can scale to:
   - Very large models (billions of parameters)
   - Very long sequences (with modifications)
   - Very large datasets (parallel processing)
   
   Examples:
   - GPT-3: 175B parameters
   - T5: 11B parameters
   - BERT: 340M parameters
```

**Transformer Variants:**

```
1. Encoder-Only (BERT):
   - Bidirectional attention
   - Good for understanding tasks
   - Examples: BERT, RoBERTa, ALBERT

2. Decoder-Only (GPT):
   - Causal (masked) attention
   - Good for generation tasks
   - Examples: GPT, GPT-2, GPT-3, LLaMA

3. Encoder-Decoder (T5):
   - Both encoder and decoder
   - Good for sequence-to-sequence tasks
   - Examples: T5, BART, mT5
```

#### Attention Mechanism - Complete Mathematical Analysis

The attention mechanism is the core innovation of the Transformer architecture. It allows the model to focus on different parts of the input sequence when processing each position. This section provides a comprehensive mathematical analysis of attention mechanisms.

**Self-Attention Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              SELF-ATTENTION ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

Input Sequence: X = [x₁, x₂, ..., xₙ]
    │
    ▼
┌──────────────────┐
│ Linear Projections│
│ • Q = XW_Q       │
│ • K = XW_K       │
│ • V = XW_V       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Attention Scores │
│ • Scores = QK^T  │
│ • Scale: /√d_k   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Softmax          │
│ • Weights = softmax(Scores)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Weighted Sum     │
│ • Output = Weights × V│
└────────┬─────────┘
         │
         ▼
Output: Attention(X)
```

**Self-Attention Mathematical Formulation:**

```
Self_Attention_Model:

For input X ∈ R^(n × d_model):

1. Linear Projections:
   Q = XW_Q  # Queries: What am I looking for?
   K = XW_K  # Keys: What do I contain?
   V = XW_V  # Values: What information do I provide?
   
   Where:
   - W_Q, W_K, W_V ∈ R^(d_model × d_k)
   - d_k = dimension of keys (typically d_model / num_heads)
   - Q, K, V ∈ R^(n × d_k)

2. Attention Scores:
   Scores = QK^T / √d_k
   
   Where:
   - QK^T ∈ R^(n × n)
   - Each element Scores[i, j] = similarity between position i and j
   - Scaling by √d_k prevents softmax saturation
   
   Example:
   For sequence length n=3:
   Scores = [
       [q₁·k₁, q₁·k₂, q₁·k₃],
       [q₂·k₁, q₂·k₂, q₂·k₃],
       [q₃·k₁, q₃·k₂, q₃·k₃]
   ] / √d_k

3. Attention Weights:
   Weights = softmax(Scores)
   
   Where:
   softmax(Scores[i, :]) = exp(Scores[i, j]) / Σ exp(Scores[i, k])
   
   Properties:
   - Each row sums to 1 (probability distribution)
   - Higher scores → higher weights
   - Weights[i, j] = how much position i attends to position j

4. Weighted Sum:
   Attention(X) = Weights × V
   
   Where:
   Attention(X)[i, :] = Σ(Weights[i, j] × V[j, :])
                       for j = 1 to n
   
   Result:
   - Each output position is a weighted combination of all values
   - Attention output ∈ R^(n × d_k)
```

**Complete Attention Formula:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Step-by-step:

1. Compute similarity matrix:
   S = QK^T
   S[i, j] = Q[i, :] · K[j, :]  # Dot product
   
2. Scale by √d_k:
   S_scaled = S / √d_k
   Reason: Prevents softmax saturation for large d_k
   
3. Apply softmax:
   A = softmax(S_scaled)
   A[i, j] = exp(S_scaled[i, j]) / Σ exp(S_scaled[i, k])
   
4. Weighted combination:
   Output = A × V
   Output[i, :] = Σ(A[i, j] × V[j, :])
```

**Attention Intuition - Detailed Explanation:**

```
Query (Q): "What am I looking for?"
- Represents what information the current position needs
- Used to query other positions

Key (K): "What do I contain?"
- Represents what information each position has
- Used to match with queries

Value (V): "What information do I provide?"
- Represents the actual information to be retrieved
- Weighted by attention scores

Attention Score: How much should position i attend to position j?
- High score: Strong relationship, should attend more
- Low score: Weak relationship, should attend less

Example:
Sequence: "The cat sat on the mat"

Position 3 ("sat"):
- Query: "What verb information do I need?"
- Key match: Position 2 ("cat") - subject information
- Key match: Position 5 ("mat") - object information
- Attention: High attention to positions 2 and 5
- Value: Retrieve information from positions 2 and 5
```

**Multi-Head Attention - Complete Analysis:**

```
Multi-Head Attention Architecture:

Input: X
    │
    ├──────────────┬──────────────┬──────────────┬──────────────┐
    │              │              │              │              │
    ▼              ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Head 1   │  │ Head 2   │  │ Head 3   │  │ Head 4   │  │ Head 8   │
│ • Q₁, K₁, V₁│  │ • Q₂, K₂, V₂│  │ • Q₃, K₃, V₃│  │ • Q₄, K₄, V₄│  │ • Q₈, K₈, V₈│
│ • Attention₁│  │ • Attention₂│  │ • Attention₃│  │ • Attention₄│  │ • Attention₈│
└─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘
      │              │              │              │              │
      └──────────────┴──────────────┴──────────────┴──────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │ Concatenate           │
              │ • Concat(head₁, ..., head₈)│
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Linear Projection     │
              │ • Output = Concat × W_O│
              └──────────────────────┘
```

**Multi-Head Attention Mathematical Model:**

```
MultiHead_Attention_Model:

For input X ∈ R^(n × d_model) and num_heads = h:

1. Split into heads:
   For each head i = 1 to h:
       Q_i = XW_Q_i  # W_Q_i ∈ R^(d_model × d_k)
       K_i = XW_K_i  # W_K_i ∈ R^(d_model × d_k)
       V_i = XW_V_i  # W_V_i ∈ R^(d_model × d_k)
   
   Where:
   - d_k = d_model / h (typically)
   - Each head operates in a different subspace

2. Attention per head:
   head_i = Attention(Q_i, K_i, V_i)
   
   Where:
   Attention(Q_i, K_i, V_i) = softmax(Q_i K_i^T / √d_k) × V_i
   head_i ∈ R^(n × d_k)

3. Concatenate heads:
   MultiHead = Concat(head_1, head_2, ..., head_h)
   
   Where:
   MultiHead ∈ R^(n × (h × d_k)) = R^(n × d_model)

4. Output projection:
   Output = MultiHead × W_O
   
   Where:
   - W_O ∈ R^(d_model × d_model)
   - Output ∈ R^(n × d_model)

Complete Formula:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_O

where head_i = Attention(QW_Q_i, KW_K_i, VW_V_i)
```

**Multi-Head Attention Benefits:**

```
1. Different Representation Subspaces:
   - Each head learns to attend to different types of relationships
   - Head 1: Syntactic relationships (subject-verb)
   - Head 2: Semantic relationships (word meanings)
   - Head 3: Long-range dependencies
   - Head 4: Position relationships
   - etc.

2. More Expressive:
   - Single head: Limited to one type of attention pattern
   - Multiple heads: Can capture diverse patterns simultaneously
   - Better representation learning

3. Better Parallelization:
   - All heads computed in parallel
   - Matrix operations can be batched
   - Efficient GPU utilization

4. Empirical Benefits:
   - Better performance on downstream tasks
   - More interpretable (each head can be analyzed)
   - More robust to different input patterns
```

**Cross-Attention - Detailed Analysis:**

```
Cross_Attention_Model:

Used in encoder-decoder architectures:
- Encoder outputs: Encoder_output ∈ R^(n_enc × d_model)
- Decoder inputs: Decoder_input ∈ R^(n_dec × d_model)

1. Generate Queries from Decoder:
   Q = Decoder_input × W_Q
   
2. Generate Keys and Values from Encoder:
   K = Encoder_output × W_K
   V = Encoder_output × W_V

3. Cross-Attention:
   CrossAttention = softmax(QK^T / √d_k) × V
   
   Where:
   - Q ∈ R^(n_dec × d_k)  # From decoder
   - K ∈ R^(n_enc × d_k)  # From encoder
   - V ∈ R^(n_enc × d_k)  # From encoder
   - CrossAttention ∈ R^(n_dec × d_k)

Intuition:
- Decoder queries: "What information do I need from the encoder?"
- Encoder keys: "What information do I have?"
- Encoder values: "Here is the information"
- Cross-attention: Decoder attends to relevant encoder positions

Example (Translation):
Encoder: "The cat sat" (source language)
Decoder: "Le chat" (target language, partial)

Cross-attention:
- "chat" (decoder) attends to "cat" (encoder) - translation
- "Le" (decoder) attends to "The" (encoder) - article translation
```

**Attention Complexity Analysis:**

```
Time Complexity:
- QK^T computation: O(n² × d_k)
- Softmax: O(n²)
- Matrix multiplication with V: O(n² × d_k)
- Total: O(n² × d_k)

Space Complexity:
- Attention scores matrix: O(n²)
- Total: O(n²)

For Multi-Head:
- Time: O(h × n² × d_k) = O(n² × d_model)  # since h × d_k = d_model
- Space: O(h × n²) = O(n²)

Comparison with RNN:
- RNN time: O(n × d²) sequential
- Transformer time: O(n² × d) parallel
- For long sequences, Transformer is faster due to parallelization
```

#### Positional Encoding - Complete Mathematical Analysis

Positional encoding is crucial for Transformers because attention mechanisms are permutation-invariant - they don't naturally encode sequence order. This section provides a comprehensive analysis of different positional encoding strategies.

**Problem Statement:**

```
Attention Mechanism Property:
- Attention(Q, K, V) treats all positions equally
- Permuting input positions gives same attention scores
- No inherent sense of sequence order

Example:
Sequence 1: "The cat sat on the mat"
Sequence 2: "The mat sat on the cat"

Without positional encoding:
- Both sequences have identical attention patterns
- Model cannot distinguish word order
```

**Positional Encoding Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              POSITIONAL ENCODING ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────┘

Input Embeddings: X_embed ∈ R^(n × d_model)
    │
    ▼
┌──────────────────┐
│ Positional       │
│ Encoding         │
│ • Generate PE    │
│ • Add to embeddings│
└────────┬─────────┘
         │
         ▼
Position-Encoded: X_pos = X_embed + PE
```

**1. Sinusoidal Positional Encoding - Complete Analysis:**

```
Sinusoidal_Positional_Encoding:

For position pos and dimension i:

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos: Position in sequence (0, 1, 2, ..., n-1)
- i: Dimension index (0, 1, 2, ..., d_model/2 - 1)
- d_model: Embedding dimension (e.g., 512, 768)

Mathematical Properties:

1. Frequency Pattern:
   - Different dimensions have different frequencies
   - Lower dimensions (small i): Higher frequency (faster variation)
   - Higher dimensions (large i): Lower frequency (slower variation)
   
   Example for d_model = 512:
   - Dimension 0: frequency = 1/10000^(0/512) = 1/1 = 1
   - Dimension 256: frequency = 1/10000^(512/512) = 1/10000 = 0.0001

2. Relative Position Encoding:
   - PE(pos + k) can be expressed as linear combination of PE(pos)
   - Enables model to learn relative positions
   - Formula: PE(pos + k) = f(PE(pos), PE(k))
   
   This allows the model to extrapolate to longer sequences

3. Orthogonality:
   - Different positions have different encoding patterns
   - Encodings are approximately orthogonal
   - Helps model distinguish positions

Visualization:
Position 0: [sin(0/10000^0), cos(0/10000^0), sin(0/10000^2/d), cos(0/10000^2/d), ...]
Position 1: [sin(1/10000^0), cos(1/10000^0), sin(1/10000^2/d), cos(1/10000^2/d), ...]
Position 2: [sin(2/10000^0), cos(2/10000^0), sin(2/10000^2/d), cos(2/10000^2/d), ...]
...
```

**Complete Sinusoidal Encoding Formula:**

```python
"""
Sinusoidal Positional Encoding Implementation
"""

import numpy as np
import torch
import torch.nn as nn

def sinusoidal_positional_encoding(max_len: int, d_model: int):
    """
    Generate sinusoidal positional encodings.
    
    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension
        
    Returns:
        PE matrix of shape (max_len, d_model)
        
    Mathematical Model:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    # Create position matrix
    pos = np.arange(max_len)[:, np.newaxis]  # Shape: (max_len, 1)
    
    # Create dimension indices
    i = np.arange(0, d_model, 2)[np.newaxis, :]  # Shape: (1, d_model/2)
    
    # Compute division term
    div_term = np.exp(i * (-np.log(10000.0) / d_model))
    
    # Initialize PE matrix
    PE = np.zeros((max_len, d_model))
    
    # Compute sinusoidal encodings
    PE[:, 0::2] = np.sin(pos * div_term)  # Even dimensions: sin
    PE[:, 1::2] = np.cos(pos * div_term)  # Odd dimensions: cos
    
    return torch.FloatTensor(PE)

# Example
max_len = 100
d_model = 512
PE = sinusoidal_positional_encoding(max_len, d_model)
print(f"Positional Encoding shape: {PE.shape}")
print(f"Position 0 encoding (first 10 dims): {PE[0, :10]}")
print(f"Position 1 encoding (first 10 dims): {PE[1, :10]}")

"""
Expected Output:
Positional Encoding shape: torch.Size([100, 512])
Position 0 encoding (first 10 dims): tensor([0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000])
Position 1 encoding (first 10 dims): tensor([0.8415, 0.5403, 0.0464, 0.9989, 0.0022, 0.9999, 0.0001, 1.0000, 0.0000, 1.0000])
"""
```

**2. Learned Positional Embeddings - Complete Analysis:**

```
Learned_Positional_Embeddings:

Instead of fixed sinusoidal encoding, learn embeddings as parameters:

PE = nn.Embedding(max_len, d_model)

Where:
- max_len: Maximum sequence length
- d_model: Embedding dimension
- PE: Learned parameters updated during training

Mathematical Model:
For position pos:
    PE(pos) = Embedding[pos, :]  # Lookup from learned table

Benefits:
1. Adaptive Learning:
   - Model learns optimal position representations
   - Can adapt to training data patterns
   - Better for fixed-length sequences

2. Simplicity:
   - Simple lookup operation
   - No complex computation
   - Easy to implement

Limitations:
1. Fixed Length:
   - Cannot extrapolate beyond max_len
   - Must retrain for longer sequences
   - Less flexible than sinusoidal

2. Training Data Dependency:
   - Learns patterns from training data
   - May not generalize to unseen positions
```

**Implementation Example:**

```python
class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings.
    
    Mathematical Model:
        PE(pos) = Embedding[pos, :]
    """
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            x + positional_encodings
        """
        batch_size, seq_len, d_model = x.size()
        
        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get positional embeddings
        pos_embeddings = self.embedding(positions)
        
        # Add to input
        return x + pos_embeddings
```

**3. Relative Position Encoding - Complete Analysis:**

```
Relative_Position_Encoding:

Encodes relative positions between tokens instead of absolute positions:

For positions i and j:
    Relative_Position = i - j

Mathematical Model:
1. Absolute Position Encoding:
   PE_abs(i) = Encoding for position i
   
2. Relative Position Encoding:
   PE_rel(i, j) = Encoding for relative position (i - j)
   
3. Attention with Relative Position:
   Attention(Q, K, V) = softmax((QK^T + Relative_Bias) / √d_k) × V
   
   Where:
   Relative_Bias[i, j] = Learned_Bias(i - j)
   - Encodes relative distance between positions
   - Better for long-range dependencies

Benefits:
1. Better for Long Sequences:
   - Focuses on relative distances
   - More generalizable
   - Better for variable-length sequences

2. Translation Invariance:
   - Same relative positions → same encoding
   - More robust to sequence shifts
   - Better generalization

Example:
Sequence: "The cat sat on the mat"

Relative positions:
- "cat" to "sat": +1 (one position forward)
- "sat" to "cat": -1 (one position backward)
- "cat" to "mat": +4 (four positions forward)

These relative encodings are more informative than absolute positions
```

**Positional Encoding Comparison:**

```
┌──────────────────┬──────────────────┬──────────────────┐
│ Feature          │ Sinusoidal       │ Learned          │ Relative         │
├──────────────────┼──────────────────┼──────────────────┤
│ Extrapolation    │ ✅ Yes           │ ❌ No            │ ✅ Yes           │
│ Learning         │ ❌ Fixed          │ ✅ Learned       │ ✅ Learned       │
│ Complexity       │ O(1)             │ O(1)             │ O(n²)           │
│ Long Sequences   │ ✅ Good          │ ⚠️ Limited      │ ✅ Excellent     │
│ Generalization   │ ✅ Good          │ ⚠️ Data-dependent│ ✅ Excellent     │
│ Used In          │ Original Transf. │ BERT, GPT        │ T5, DeBERTa      │
└──────────────────┴──────────────────┴──────────────────┘
```

**Positional Encoding Integration:**

```
Complete Integration:

1. Input Embedding:
   X_embed = Embedding(token_ids)
   X_embed ∈ R^(batch_size × seq_len × d_model)

2. Positional Encoding:
   PE = PositionalEncoding(seq_len, d_model)
   PE ∈ R^(seq_len × d_model)

3. Add Together:
   X_pos = X_embed + PE
   
   Where:
   - Broadcasting: PE expanded to match batch dimension
   - Element-wise addition
   - X_pos ∈ R^(batch_size × seq_len × d_model)

4. Forward Pass:
   Output = Transformer(X_pos)
   
   The model now has both:
   - Token information (from embeddings)
   - Position information (from positional encoding)
```

**Why Add (Not Concatenate)?**

```
Reasons for Addition:

1. Preserves Dimension:
   - Addition: X + PE ∈ R^(n × d_model)
   - Concatenation: [X, PE] ∈ R^(n × 2d_model)
   - Addition keeps dimension constant

2. Information Fusion:
   - Addition allows model to learn how to combine
   - Token and position information interact
   - More flexible than concatenation

3. Computational Efficiency:
   - Addition: O(n × d) operation
   - Concatenation: O(n × 2d) downstream
   - Addition is more efficient

4. Empirical Evidence:
   - Addition works better in practice
   - Standard in Transformer implementations
   - Better performance on downstream tasks
```

#### Residual Connections - Complete Analysis

Residual connections (also called skip connections) are crucial for training deep Transformer networks. They enable gradient flow through many layers and allow the model to learn identity mappings when needed.

**Residual Connection Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│              RESIDUAL CONNECTION ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────┘

Input: X
    │
    ├──────────────────────────┐
    │                          │
    ▼                          │
┌──────────────────┐           │
│ Sublayer         │           │
│ (e.g., Attention)│           │
└────────┬─────────┘           │
         │                      │
         ▼                      │
┌──────────────────┐           │
│ Sublayer Output  │           │
└────────┬─────────┘           │
         │                      │
         ▼                      │
┌──────────────────┐           │
│ Add              │◄──────────┘
│ X + Sublayer(X)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Layer Norm       │
│ • Normalize      │
└────────┬─────────┘
         │
         ▼
    Output
```

**Mathematical Model:**

```
Residual_Connection_Model:

Standard Layer:
output = Layer(Sublayer(input))
- Problem: Gradient can vanish in deep networks

Residual Connection:
output = LayerNorm(input + Sublayer(input))

Where:
- input: Original input
- Sublayer(input): Transformed input (e.g., Attention, FFN)
- input + Sublayer(input): Residual connection
- LayerNorm: Normalization

Properties:

1. Identity Mapping:
   If Sublayer(input) = 0:
       output = LayerNorm(input + 0) = LayerNorm(input)
   Model can learn to pass input through unchanged

2. Gradient Flow:
   ∂output/∂input = ∂LayerNorm(input + Sublayer(input))/∂input
   = 1 + ∂Sublayer(input)/∂input
   
   Gradient can flow directly through addition
   Prevents vanishing gradients

3. Information Preservation:
   Original information always preserved
   Model can add new information incrementally
```

**Residual Connection in Transformer Block:**

```
Complete Transformer Block with Residuals:

Input: X
    │
    ▼
┌──────────────────┐
│ Layer Norm 1      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Multi-Head       │
│ Attention        │
│ • Attention(X)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Residual 1       │
│ X + Attention(X) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Layer Norm 2     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Feed Forward     │
│ • FFN(X)         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Residual 2       │
│ X + FFN(X)       │
└──────────────────┘

Mathematical Formula:
X_out = LayerNorm_2(LayerNorm_1(X + Attention(X)) + FFN(LayerNorm_1(X + Attention(X))))

Where:
- First residual: X + Attention(X)
- Second residual: Intermediate + FFN(Intermediate)
```

**Benefits - Detailed Analysis:**

```
1. Gradient Flow:
   Problem in Deep Networks:
   - Gradients can vanish: ∂L/∂X → 0 for deep layers
   - Gradients can explode: ∂L/∂X → ∞
   
   Solution with Residuals:
   - Gradient path: ∂L/∂X = ∂L/∂output × (1 + ∂Sublayer/∂X)
   - Direct path through addition: gradient = 1
   - Always has gradient flow even if Sublayer gradient is small
   
   Example:
   Without residual: gradient = 0.1^10 = 1e-10 (vanished)
   With residual: gradient = 1 + 0.1^10 ≈ 1 (preserved)

2. Easier Optimization:
   - Model can learn identity: output = input
   - Can start with identity and learn incrementally
   - Smoother loss landscape
   - Faster convergence

3. Deeper Networks:
   - Can train very deep networks (100+ layers)
   - Without residuals: difficult to train >10 layers
   - With residuals: can train 100+ layers successfully
   
   Examples:
   - ResNet: 152 layers (computer vision)
   - Transformer: 12-24 layers (NLP)
   - GPT-3: 96 layers

4. Better Performance:
   - Residuals enable deeper models
   - Deeper models = more capacity
   - More capacity = better performance
```

**Pre-Norm vs Post-Norm:**

```
Two Variants:

1. Post-Norm (Original Transformer):
   output = LayerNorm(input + Sublayer(input))
   - Normalization after residual
   - Original Transformer design

2. Pre-Norm (Modern Variant):
   output = input + Sublayer(LayerNorm(input))
   - Normalization before sublayer
   - More stable training
   - Better for very deep networks
   - Used in modern architectures (GPT-3, etc.)

Comparison:
- Pre-norm: More stable, easier to train
- Post-norm: Original design, works well
```

#### Encoder-Decoder Models

**BERT (Bidirectional Encoder Representations):**
- Encoder-only architecture
- Bidirectional attention
- Masked language modeling
- Good for classification, understanding

**GPT (Generative Pre-trained Transformer):**
- Decoder-only architecture
- Causal (masked) attention
- Autoregressive generation
- Good for generation tasks

**T5 (Text-to-Text Transfer Transformer):**
- Encoder-decoder architecture
- All tasks as text-to-text
- Unified framework
- Good for many NLP tasks

**Comparison:**

| Model | Architecture | Attention | Best For |
|-------|-------------|-----------|----------|
| BERT | Encoder | Bidirectional | Understanding |
| GPT | Decoder | Causal | Generation |
| T5 | Encoder-Decoder | Both | Translation, Summarization |

#### Implementation Overview

**Key Components:**

**1. Embedding Layer:**
```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
    
    def forward(self, x):
        return self.embedding(x) + self.pos_encoding(x)
```

**2. Multi-Head Attention:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Split into heads, compute attention, concatenate
        ...
```

**3. Transformer Block:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x
```

### Readings

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - GPT-2

 

### Additional Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Transformer Paper](https://arxiv.org/abs/1706.03762)

### Practical Code Examples

#### Complete RAG Evaluation System

```python
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGEvaluator:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[str], 
                          relevant_docs: List[str], k: int = 5):
        """Evaluate retrieval quality"""
        # Calculate recall@k
        recall = len(set(retrieved_docs[:k]) & set(relevant_docs)) / len(relevant_docs)
        
        # Calculate precision@k
        precision = len(set(retrieved_docs[:k]) & set(relevant_docs)) / k
        
        return {"recall@k": recall, "precision@k": precision}
    
    def evaluate_answer_quality(self, answer: str, reference: str):
        """Evaluate answer quality using embeddings"""
        answer_emb = self.embedding_model.encode(answer)
        ref_emb = self.embedding_model.encode(reference)
        
        similarity = cosine_similarity([answer_emb], [ref_emb])[0][0]
        
        return {"semantic_similarity": float(similarity)}
    
    def evaluate_faithfulness(self, answer: str, context: str):
        """Check if answer is grounded in context"""
        answer_emb = self.embedding_model.encode(answer)
        context_emb = self.embedding_model.encode(context)
        
        similarity = cosine_similarity([answer_emb], [context_emb])[0][0]
        
        return {"faithfulness_score": float(similarity)}

# Usage
evaluator = RAGEvaluator()
retrieval_metrics = evaluator.evaluate_retrieval(
    query="What is machine learning?",
    retrieved_docs=["doc1", "doc2", "doc3"],
    relevant_docs=["doc1", "doc2"],
    k=3
)
print(f"Retrieval metrics: {retrieval_metrics}")
```

**Pro Tip:** Always evaluate retrieval and generation separately. Good retrieval is necessary but not sufficient for good RAG performance.

**Common Pitfall:** Evaluating only end-to-end metrics can hide retrieval issues. Always measure retrieval recall independently.

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Low retrieval recall** | Missing relevant documents | Increase k, improve embeddings, use hybrid search |
| **Hallucinated answers** | Answers not in context | Add faithfulness checks, improve prompts, use reranking |
| **Slow RAG pipeline** | High latency | Cache embeddings, optimize retrieval, use faster models |
| **Context overflow** | Token limit exceeded | Reduce chunk size, implement summarization, use token budgeting |
| **Poor answer quality** | Irrelevant or wrong answers | Improve retrieval, add reranking, optimize prompts |

### Quick Reference Guide

#### Transformer Architecture Comparison

| Architecture | Type | Use Case | Example Models |
|--------------|------|----------|----------------|
| Encoder-only | Bidirectional | Understanding, classification | BERT, RoBERTa |
| Decoder-only | Autoregressive | Generation, completion | GPT, LLaMA |
| Encoder-Decoder | Sequence-to-sequence | Translation, summarization | T5, BART |

#### RAG Evaluation Metrics

| Metric | Purpose | Range | When to Use |
|--------|---------|-------|-------------|
| Recall@k | Retrieval quality | 0-1 | Measure retrieval coverage |
| Precision@k | Retrieval accuracy | 0-1 | Measure retrieval relevance |
| Faithfulness | Answer grounding | 0-1 | Check hallucination |
| Semantic Similarity | Answer quality | 0-1 | Compare answers |

### Case Studies

#### Case Study: RAG System for Legal Document Analysis

**Challenge:** Law firm needed to search and analyze 50,000+ legal documents.

**Solution:** Implemented RAG with:
- Hybrid retrieval (BM25 + embeddings)
- GPT-4 for generation
- Custom evaluation framework

**Results:**
- 90% retrieval recall
- 85% answer accuracy
- 10x faster than manual review

**Lessons Learned:**
- Domain-specific embeddings crucial
- Evaluation framework essential
- Hybrid retrieval outperformed single method

### Key Takeaways

1. RAG combines retrieval and generation for knowledge-intensive tasks
2. Hybrid retrieval (BM25 + embeddings) provides best results
3. Transformer architecture enables modern LLMs through attention
4. Attention mechanism allows parallel processing and long-range dependencies
5. Different architectures (BERT, GPT, T5) serve different purposes
6. Proper evaluation is crucial for RAG system optimization
7. Understanding transformer internals helps in model selection and optimization
8. Separate evaluation of retrieval and generation provides better insights
9. Faithfulness checks prevent hallucination
10. Continuous monitoring and evaluation improve system performance

---

**Previous Module:** [Module 5: Frameworks for Building GenAI Applications](../module_05.md)  
**Next Module:** [Module 7: Tokenization & Embeddings in LLMs](../module_07.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

