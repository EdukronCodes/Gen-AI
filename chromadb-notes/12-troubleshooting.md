# 12 — Troubleshooting

## Overview

This chapter provides diagnostic steps, common failure modes, and practical fixes for issues encountered while running ChromaDB-based systems: memory, query quality, ingestion, backups, and security incidents.

---

## Common issues and fixes

- High memory usage:
  - Cause: large in-memory index or storing many high-dimension vectors.
  - Fixes: enable on-disk persistence, use quantized indices (PQ/INT8), increase instance memory, or partition data.

- Poor search relevance:
  - Cause: mismatched embedding models, poor chunking, or normalization mismatches.
  - Fixes: verify embedding model/version, review chunking strategy, normalize vectors, and evaluate on a labeled set.

- Slow queries / high latency:
  - Cause: aggressive `efSearch` / probes, very large `k`, or cold caches.
  - Fixes: reduce `k`, lower `efSearch`, increase caching, or scale horizontally for read replicas.

- Ingestion failures:
  - Cause: incorrect embedding dimension, network timeouts, or invalid metadata types.
  - Fixes: validate embedding shapes before write, add retry/backoff, and log and move failed batches to dead-letter storage.

---

## Debugging checklist

1. Validate embeddings: check shapes, numeric ranges (no NaNs/inf), and expected dimension.
2. Verify item storage: query by id and inspect metadata and stored `document`.
3. Run synthetic queries: measure distances between known related and unrelated items to sanity-check embedding space.
4. Inspect logs and metrics for error patterns and resource saturation.

---

## Backup and restore issues

- If restore fails: check format compatibility, schema mismatches, and partial writes.
- Always test restore in a staging environment and validate content with sample queries.

---

## Reproducible crash handling

- Capture a minimal repro case: small dataset and sequence of operations that triggers the crash.
- Attach logs, stack traces, environment details (package versions), and input data for upstream debugging.

---

## Performance profiling tools

- Use CPU/memory profilers (perf, py-spy) to spot hotspots.
- Measure end-to-end latency with tracing (OpenTelemetry) from API entry to Chroma query and reranker.

---

## When to escalate

- Escalate to platform or upstream when you have reproducible crashes, suspected data corruption, or a security incident.
.
