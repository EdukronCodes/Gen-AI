# Project 3 — Retail Chatbot for Inventory & Order Queries (Generative AI + RAG)

## What’s inside

- `NOTES.md`: RAG design notes (ingestion → chunking → embeddings → retrieval → grounded response)
- `data/rag_retail_schema_and_seed.sql`: sample relational dataset for inventory/orders/pricing/offers
- `docs/knowledge_base.md`: sample “documents” you would index into a vector DB
- `prompts/system_prompt.md`: example system prompt template with guardrails

## Quick start (SQLite demo)

```bash
cd Project-3-Retail-RAG-Chatbot-Inventory-Orders
sqlite3 retail_rag.db < data/rag_retail_schema_and_seed.sql
```

