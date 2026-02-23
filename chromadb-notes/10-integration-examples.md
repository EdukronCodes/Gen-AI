# 10 — Integration Examples

## Overview

This chapter provides runnable integration examples showing how to connect ChromaDB to embedders, LLMs for RAG, web UIs, and deployment examples (Docker + docker-compose). Each example is intentionally minimal and runnable locally; adapt for production by adding retries, secrets management, and observability.

---

## Example 1 — Minimal Python end-to-end (runnable)

Create `examples/quickstart.py`:

```py
from chromadb import Client
from sentence_transformers import SentenceTransformer

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = Client()
    col = client.create_collection('examples_quickstart')

    docs = [
        {'id':'d1','text':'How to reset your password','meta':{'source':'help'}},
        {'id':'d2','text':'How to change your email','meta':{'source':'help'}},
    ]

    texts = [d['text'] for d in docs]
    embs = model.encode(texts).tolist()

    col.add(ids=[d['id'] for d in docs], embeddings=embs, metadatas=[d['meta'] for d in docs], documents=texts)

    q = 'I forgot my password'
    qemb = model.encode([q]).tolist()
    res = col.query(query_embeddings=qemb, n_results=2)
    print(res)

if __name__ == '__main__':
    main()
```

Run locally after installing requirements:

```bash
pip install chromadb sentence-transformers
python examples/quickstart.py
```

---

## Example 2 — RAG with an LLM (conceptual)

Flow:

1. Embed user question.
2. Retrieve top-k chunks from Chroma.
3. Build prompt with retrieved chunks and send to LLM.
4. Return LLM output to user and optionally store conversation context.

Pseudo code:

```py
qemb = embedder.encode([question])
cands = col.query(query_embeddings=qemb, n_results=10)
context = "\n\n".join([c['document'] for c in cands])
prompt = f"Use the following context to answer:\n{context}\nQuestion: {question}"
answer = llm.complete(prompt)
```

Notes: chunk length, prompt engineering, and safety checks significantly affect the quality of RAG results.

---

## Example 3 — Chatbot with conversation embeddings

- Store each user/assistant turn as an item in a `conversations` collection with `conversation_id` metadata and incremental `turn_index`.
- When generating a reply, retrieve recent turns by `conversation_id`, embed the latest question, and query for relevant past turns to use as context.

Schema example:

```
collection: conversations
metadata: {conversation_id, turn_index, role}
```

---

## Example 4 — Search UI integration (React + backend)

Pattern:

- Backend: exposes REST or GraphQL endpoint that runs query embedding and Chroma query.
- Frontend: sends user query, displays top-k results with metadata and source links.

Backend pseudo:

```py
def search(query):
    qemb = embedder.encode([query])
    res = col.query(query_embeddings=qemb, n_results=10)
    return format_for_ui(res)
```

Frontend considerations:

- Highlight matched excerpts and show provenance (source, url, created_at).
- Offer filters and faceted navigation by metadata fields.

---

## Example 5 — Docker + docker-compose for local testing

`docker-compose.yml` (minimal):

```yaml
version: '3.8'
services:
  app:
    build: .
    volumes:
      - ./:/app
    command: python examples/quickstart.py
```

Build and run:

```bash
docker compose up --build
```

---

## Connectors and best-of-breed embedders

- OpenAI: high-quality embeddings (hosted) — remember cost and rate limits.
- SentenceTransformers / Hugging Face: good local/inference options.
- Vector pipeline connectors: S3, web crawlers, DB connectors — build ETL to normalize content and metadata.

---

## Testing and validation

- Add integration tests that:
  - Insert known items and assert retrieval for sample queries.
  - Validate metadata filters and idempotency.

Example test (pytest conceptual):

```py
def test_basic_indexing():
    client = Client()
    col = client.create_collection('test_integration')
    col.add(ids=['t1'], embeddings=[[0.1,0.2,0.3]], metadatas=[{'source':'test'}], documents=['hello'])
    res = col.query(query_embeddings=[[0.1,0.2,0.3]], n_results=1)
    assert res['ids'][0][0] == 't1'
```

