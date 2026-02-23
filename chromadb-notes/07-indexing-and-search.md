# 07 — Indexing & Search

## Overview

Indexing and search are the heart of any vector database. This chapter covers index types, tuning parameters, search strategies, reranking, hybrid search (vector + lexical), caching, and relevance evaluation.

---

## Index types and selection

- Brute-force (exact): compute distances against all vectors — accurate but slow for large datasets.
- HNSW (graph): fast, high recall for many workloads — common default.
- IVF + PQ: inverted file with product quantization — memory efficient for very large corpora.

Table: index types

| Type | Speed | Memory | Accuracy | Typical use |
|---|---:|---:|---:|---|
| Brute-force | Slow | Low | Exact | Small datasets, testing |
| HNSW | Very fast | Medium–High | High | General-purpose production |
| IVF+PQ | Fast | Low | Medium | Extremely large datasets |

---

## Similarity metrics and normalization

- Cosine similarity: requires normalized vectors or use cosine implementation. Good default for text embeddings.
- Dot product: useful when vector magnitude encodes importance.
- Euclidean (L2): sometimes used for dense vector models.

Normalization note: when using cosine, it's common to L2-normalize embeddings at generation time so queries and stored vectors are comparable.

---

## Search parameters and tuning

- `k`: number of nearest neighbors to return — balance between recall and processing cost.
- Search-time probes / efSearch (HNSW): higher values increase recall but also latency.
- Index-time parameters (HNSW efConstruction, M): tradeoff between build time, memory, and search accuracy.

Guidance:

- Start with moderate `k` (10–50) then rerank down to final result set.
- Tune `efSearch` to reach desired recall while measuring latency.

---

## Hybrid and reranked search

- Hybrid search: combine vector similarity with lexical/full-text signals (BM25) to improve precision for certain query types.
- Reranking: retrieve N candidates by vector search (larger N) and rerank with more expensive signals (LLM, lexical similarity, recency boosts).

Rerank flow:

1. Query embedding → top-N vector candidates.
2. Apply lexical scoring or context-aware scoring.
3. Combine scores (linear blend or learned model).

---

## Example tuning table

| Scenario | Index | efSearch | k | Notes |
|---|---|---:|---:|---|
| Low-latency web search | HNSW | 50 | 10 | prioritize quick responses |
| High-recall QA | HNSW | 300 | 100 | higher efSearch for recall, rerank after |
| Massive archive | IVF+PQ | N/A | 200 | use PQ to reduce memory footprint |

---

## Caching and result stability

- Cache hot queries to reduce repeated computation and latency.
- Ensure result stability by versioning embedder and index parameters; store index metadata with each build.

---

## Relevance evaluation

- Metrics: recall@k, precision@k, MRR, MAP.
- Use labeled query sets to measure impact of index tuning and reranking.

---

## Practical example (query + rerank)

```py
# 1. query embedding
q_emb = embedder.encode([query])[0]
# 2. vector search for top-N
candidates = col.query(query_embeddings=[q_emb], n_results=200)
# 3. rerank candidates with lexical score or LLM
# combine scores and return top-k
```
