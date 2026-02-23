# 11 — Best Practices

## Overview

This chapter collects practical best practices for building reliable, maintainable, and high-quality systems on top of ChromaDB: schema design, ingestion, embeddings, query design, monitoring, security, and operational hygiene.

---

## Schema and metadata

- Keep metadata small, typed, and query-friendly. Avoid large blobs inside metadata.
- Use controlled vocabularies (enums) for fields like `source`, `type`, `category` to improve filter efficiency.
- Record provenance and processing metadata (`embedder_model`, `embedder_version`, `ingest_run_id`).

---

## Ingestion best practices

- Use deterministic IDs (e.g., hash of `doc_id + chunk_index`) to make ingestion idempotent.
- Batch embeddings and collection writes to improve throughput and reduce API overhead.
- Sanitize inputs, trim whitespace, and normalize unicode to avoid duplicate or near-duplicate vectors.

---

## Embeddings and models

- Pin embedding model versions to prevent silent behavior changes.
- Normalize vectors if using cosine similarity and document the normalization flag.
- Evaluate embedding models on a labeled validation set: measure recall@k and downstream task metrics.

---

## Query design and relevance

- Prefer pre-filtering by selective metadata when possible to reduce search scope and cost.
- Use reranking on larger candidate sets for precision-sensitive applications (QA, legal search).
- Cache hot query results and warm caches for predictable traffic patterns.

---

## Operational and monitoring

- Monitor key metrics: query latency (P50/P95/P99), QPS, memory usage, index size, and error rates.
- Alert on sustained increases in latency, memory pressure, or ingestion failures.
- Keep automated backups and test restores periodically.

---

## Security and compliance

- Use RBAC and separate collections or namespaces per tenant for multi-tenant setups.
- Encrypt data at rest and in transit; do not store secrets in metadata or source control.
- Implement input validation to avoid injection and malformed filter queries.

---

## Testing and validation

- Add unit tests for ingestion, indexing, and query correctness.
- Maintain a small labeled dataset for regression testing of relevance.
- Perform load testing before production rollout to validate throughput and latency.

---

## Cost optimization

- Batch embedding calls, and cache embeddings for repeated inputs.
- Consider smaller embedder models for high-volume, lower-precision needs.
- Use quantized indices after validating recall trade-offs.

---

## Documentation and runbooks

- Document collection-level configs (embedding dim, model, normalization) in a central place.
- Maintain runbooks for common failures: slow queries, out-of-memory, and restore procedures.

