# 04 — Data Model

## Overview

The data model for ChromaDB centers around compact items that combine an identifier, a high-dimensional embedding, small structured metadata, and optional human-readable content. This chapter clarifies modeling choices, metadata schema design, embedding storage formats, sharding/partitioning strategies, schema evolution, and sample schemas for common workloads.

---

## Item structure

- `id` (string): unique identifier for the item — use deterministic, human-readable IDs when possible to enable safe re-ingestion and updates.
- `embedding` (float array): dense vector representation. Typical storage uses float32; quantized formats can reduce memory footprint.
- `metadata` (dict): structured key/value data for filtering, faceting, and traceability.
- `document` (optional): the original text or a short excerpt for display and reranking.

Example JSON item:

```json
{
  "id": "doc-123_chunk-0",
  "embedding": [0.12, -0.34, 0.98, ...],
  "metadata": {"source":"web","author":"Alice","created_at":"2025-02-01"},
  "document": "First chunk of the article ..."
}
```

---

## Metadata design principles

- Keep metadata minimal and typed — avoid embedding large text inside metadata fields.
- Use small enums or controlled vocabularies for common fields (e.g., `type: article | blog | product`).
- Record provenance: `source`, `doc_id`, `ingest_run`, and processing step versions. This is essential for traceability and debugging.

Common metadata fields and suggested types:

| Field | Type | Notes |
|---|---|---|
| `source` | string | origin (web, s3, db) |
| `doc_id` | string | original document id |
| `chunk_index` | int | index in chunking pipeline |
| `language` | string | language code (en, fr, etc.) |
| `confidence` | float | optional model confidence |

---

## Embedding storage formats and optimization

- Float32: standard, simple and widely compatible.
- FP16 / bfloat16: reduces memory usage with small precision loss; requires library support.
- Quantization (INT8, PQ): large memory savings at the cost of accuracy; requires careful evaluation.
- Compression: store embeddings compressed for cold archives (e.g., gzip, zstd) and reload when needed.

Performance note: quantized indices may require conversion or approximation during query-time; test end-to-end impact on recall.

---

## Schema examples

1. Document QA collection:

```
collection: docs_qa
embedding_dim: 1536
metadata: {source:string, doc_id:string, chunk_index:int, created_at:datetime}
```

2. Product recommendations:

```
collection: products
embedding_dim: 768
metadata: {category:string, price:number, in_stock:bool}
```

---

## Sharding and partitioning strategies

- Logical partitioning: split by tenant, dataset, or time window by creating multiple collections.
- Application-level sharding: routing layer decides which collection(s) to query based on metadata.
- External sharding: use multiple nodes with an orchestrator and route queries to appropriate nodes.

Partitioning patterns:

- By tenant: `collection_tenant_{tenant_id}` — isolates tenant data and simplifies access control.
- By time window: `collection_docs_2025_Q1` — useful for lifecycle policies and archiving.

---

## Schema evolution and migration

- Adding fields: compatible; add new metadata fields and backfill when convenient.
- Changing embedding model or dimension: requires a reindex (create new collection and re-ingest).

Migration checklist:

1. Export current metadata and IDs.
2. Recompute embeddings with new model.
3. Insert into new collection, validate with test queries.
4. Switch production reads to the new collection and retire the old one.

---

## Referential data and external storage

- Store large binary objects (PDFs, images) externally (S3, blob storage) and reference them via metadata URLs.
- Keep `document` fields short (summaries or excerpts) to reduce storage and bandwidth.

Example metadata for external storage:

```json
"metadata": {"source":"s3","s3_key":"bucket/path/doc-123.pdf","summary":"First paragraph..."}
```

---

## Validation and schema enforcement

- Validate item shapes before ingest (embedding length, id uniqueness, metadata types).
- Use schema validators locally (pydantic, jsonschema) to catch issues before writing to collection.

Example pydantic model (conceptual):

```py
from pydantic import BaseModel
from typing import List, Dict

class Item(BaseModel):
    id: str
    embedding: List[float]
    metadata: Dict[str, object]
    document: str | None = None

    def validate_dim(self, expected_dim: int):
        assert len(self.embedding) == expected_dim
```

---

## Practical tips

- Store the embedder model name and version in metadata or collection-level config for reproducibility.
- Use deterministic ids derived from document identifiers and chunk indices.
- Keep index-mutable metadata small to avoid excessive reindexing costs.