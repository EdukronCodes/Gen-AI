# Retail AI Projects — Notes + Sample Datasets

This workspace contains **3 retail-focused projects**, each with:

- **`NOTES.md`**: full notes (architecture, implementation details, KPIs, guardrails, evaluation)
- **`data/`**: sample datasets you can load into SQL / analytics tools

## Projects

1. **Agentic AI Retail Operations Assistant (Multi-Agent Intelligent System)**  
   Folder: `Project-1-Agentic-AI-Retail-Operations-Assistant/`

2. **Customer Segmentation for Personalized Marketing (Clustering)**  
   Folder: `Project-2-Customer-Segmentation-Personalized-Marketing/`

3. **Retail Chatbot for Inventory and Order Queries (Generative AI + RAG)**  
   Folder: `Project-3-Retail-RAG-Chatbot-Inventory-Orders/`

## How to use the datasets

- For projects with a `*.sql` seed file:
  - Load into **SQLite** (quick local demo) or port to Postgres/MS SQL.
  - Example (SQLite):

```bash
sqlite3 retail_ops.db < Project-1-Agentic-AI-Retail-Operations-Assistant/data/retail_ops_schema_and_seed.sql
```

```bash
sqlite3 retail_rag.db < Project-3-Retail-RAG-Chatbot-Inventory-Orders/data/rag_retail_schema_and_seed.sql
```

- For projects with `*.csv`:
  - Open directly in Excel/Tableau, or load with Python/Pandas, or import into SQL.

