# 05 — Embeddings

## Overview

Embeddings are the numeric backbone of semantic search — fixed-length dense vectors that encode semantics of text, images, or other modalities. This chapter covers model selection, generation strategies, normalization, batching, storage formats, evaluation, and cost/performance trade-offs.

---

## Embedding model selection

- Small models: faster and cheaper, suitable for prototyping and some high-throughput tasks.
- Large models: better semantic quality and robustness; more compute and cost.

Criteria for choosing a model:

- Task accuracy requirements (QA, search relevance, recommendations).
- Throughput and latency constraints.
- Cost budget for inference.
- Availability of on-device or hosted inferencing.

Table: model selection trade-offs

| Model class | Accuracy | Latency | Cost | Use cases |
|---|---:|---:|---:|---|
| Small | Low–Medium | Low | Low | Prototyping, large-scale cheap indexing |
| Medium | Medium | Medium | Medium | Search and RAG for many apps |
| Large | High | High | High | High-precision QA and critical applications |

---

## Generating embeddings

- Batch embeddings to maximize throughput and reduce API call overhead.
- Normalize embeddings if using cosine similarity (unit length vectors).
- Record the embedder model, parameters, and timestamp in metadata for reproducibility.

Example Python pseudocode for batching:

```py
def batch_embed(texts, model, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch)
        embeddings.extend(embs)
    return embeddings
```

---

## Normalization and similarity

- For cosine similarity, normalize vectors to unit L2 norm:

$$\hat{v} = \frac{v}{\|v\|_2}$$

- If vectors are normalized, cosine similarity becomes simple dot product between normalized vectors.

---

## Storage formats and optimizations

- Float32: standard, straightforward.
- FP16 / bfloat16: reduces memory with minor precision loss.
- Quantization (INT8, PQ): large memory savings at the cost of some accuracy.

When to quantize:

- Use quantization for large datasets where memory is the bottleneck; run experiments to measure recall drop.

---

## Evaluation of embedding quality

- Intrinsic tests: nearest-neighbor recall on held-out labeled pairs.
- Extrinsic tests: downstream task performance (search relevance, QA accuracy).

Suggested evaluation workflow:

1. Prepare a labeled validation set (query → expected relevant doc ids).
2. Index a sample corpus and run queries.
3. Compute recall@k, MAP, or MRR to compare embedding models.

---

## Cost and throughput

- Use batching and larger instances for faster throughput.
- Monitor cost per 1k embeddings; consider caching repeated embeddings.

---

## Practical tips and pitfalls

- Keep consistent embedding dimension across a collection.
- Store model metadata (`embedder_model`, `embedder_version`, `normalize`) on each item or collection-level config.
- Beware of silent changes in upstream embedding models (version updates) — always pin versions.

