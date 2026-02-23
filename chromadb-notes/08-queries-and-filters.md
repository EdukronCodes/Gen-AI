# 08 — Queries & Filters

## Overview

This chapter covers how to query ChromaDB effectively: query types, filter syntax and strategies, hybrid search patterns, reranking approaches, evaluation metrics for query relevance, and worked examples with code and diagrams.

---

## Query types and semantics

- Similarity query: provide an embedding (or text to be embedded) and retrieve top-k nearest vectors.
- ID lookup: fetch one or multiple items by id.
- Filtered query: apply metadata constraints in combination with vector search.
- Hybrid query: combine vector scores with lexical/fuzzy matching to increase precision.

Query examples:

- Vector-only: find semantically similar items.
- Filtered vector: find similar items within `source='policy'` and `language='en'`.
- Hybrid: prefer exact matches for entity names while using vector similarity for broader context.

---

## Filters: design, performance, and security

Design principles:

- Keep metadata small and typed to make filters efficient.
- Prefer selective filters early in the pipeline to reduce candidate sets.
- Whitelist allowed filter fields and validate types to prevent misuse.

Filter operations:

- Equality: `{ "source": "web" }`
- Range: `{ "created_at": { "$gte": "2024-01-01" } }`
- Membership: `{ "category": { "$in": ["faq","guide"] } }`
- Boolean composition: `$and`, `$or`, `$not`

Performance tips:

- Index or pre-compute facets for frequently used filter fields when possible.
- Avoid very high-cardinality filters that undo the benefits of vector indexing.

Security:

- Validate and sanitize all filter inputs; treat them as application data, not code.
- Enforce a schema or use a server-side filter builder to avoid injection or malformed queries.

---

## Hybrid search strategies

Three common hybrid patterns:

1. Pre-filter then vector search: apply restrictive metadata filters first, then vector search on resulting subset.
2. Vector search then post-filter: run vector retrieval globally and apply filters to the returned candidates.
3. Score fusion: compute vector and lexical scores for candidates and combine using a weighted blend or learned reranker.

Choosing a pattern:

- Use pre-filter when metadata is highly selective and reduces work significantly.
- Use post-filter if filters are orthogonal and you want to preserve vector ranking.
- Use score fusion when both lexical match quality and semantic similarity matter.

---

## Reranking and candidate selection

Typical approach:

1. Retrieve a larger candidate set (N) via vector search (e.g., N = 100–500).
2. Apply more expensive rerankers on candidates (BM25, LLM-based semantic reranker, or application-specific heuristics).
3. Select final top-k for presentation.

Benefits:

- Efficient: vector search provides a fast narrow-down; reranker adds precision only where needed.
- Flexible: rerankers can combine recency, popularity, or metadata signals.

Mermaid: rerank pipeline

```mermaid
flowchart LR
  Q[User Query] --> QE[Query Embedder]
  QE --> DB[Vector Search (top-N)]
  DB --> RR[Reranker (BM25/LLM/Heuristics)]
  RR --> Out[Top-k Results]
```

Score blending formula (linear):

$$score = \alpha \cdot score_{vector} + (1-\alpha) \cdot score_{lexical}$$

Tune \alpha using a labeled validation set.

---

## Pagination, stability, and tie-breakers

- Implement stable ordering (e.g., tie-break by `created_at` or `id`) when paginating results.
- Use cursor-based pagination when returning large result sets to avoid repeated computation.

---

## Evaluation and metrics

- Use labeled queries and metrics to assess improvements: recall@k, precision@k, MRR, MAP.
- Run offline experiments before deploying scoring changes; use A/B tests when possible.

---

## Worked Python example (filtered query + rerank)

```py
from chromadb import Client
client = Client()
col = client.get_collection('docs_qa')

query_text = "How do I reset my password?"
q_emb = embedder.encode([query_text])[0]

# Vector search with metadata filter
candidates = col.query(query_embeddings=[q_emb], n_results=200, where={"source":"help_center","language":"en"})

# Pseudo BM25 scoring on candidate texts
texts = [c['document'] for c in candidates]
bm25_scores = bm25.score(texts, query_text)

# Combine and pick top-k
combined = []
for i, c in enumerate(candidates):
    vscore = 1 - c['distance']  # if distance is returned; convert to similarity
    lscore = bm25_scores[i]
    score = 0.7 * vscore + 0.3 * lscore
    combined.append((score, c))

topk = [c for _, c in sorted(combined, key=lambda x: x[0], reverse=True)][:5]
```

---

## Complex filters and examples

Example complex filter combining boolean logic:

```json
{
  "$and": [
    {"source": "web"},
    {"language": "en"},
    {"$or": [{"category": "faq"}, {"category": "guide"}]}
  ]
}
```

Performance note: test whether applying the `$and` first yields better latency than post-filtering candidates.

---

## Operational considerations

- Whitelist and validate fields used in filters.
- Monitor filter cardinality and query distribution to detect hotspots.
- Cache expensive rerank results and warm caches for expected traffic.

---
