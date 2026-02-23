# 02 — Installation

## Overview

This chapter covers installing ChromaDB, setting up environments for reproducible deployments, optional persistence backends, server and containerized deployments, and common troubleshooting steps. Examples focus on Python but include notes for JavaScript usage where applicable.

---

## Quick installs

Python (recommended for examples in this repo):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install chromadb
```

Node.js / JavaScript (client-only usage):

```bash
# (example; check official SDK for exact package name)
npm install chromadb
```

---

## Installation variants and extras

- Minimal: `pip install chromadb` — good for local prototyping.
- Full environment: include optional dependencies for persistence layers, serialization, and utilities (e.g., `chromadb[persistence]` if such extras exist in your version).
- Docker: use a containerized environment for consistent deployments.

Docker example (base image for an app that uses Chroma client locally):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "app.py"]
```

---

## Environment recommendations

- Python: 3.9+ recommended. Use virtual environments (`venv`, `virtualenv`, or `poetry`).
- Pin package versions in `requirements.txt` or `pyproject.toml` to ensure reproducibility.
- For production, prefer containerized builds with pinned dependencies and minimal base images.

Example `requirements.txt` snippet:

```
chromadb==1.6.0
openai==1.0.0
sentence-transformers==2.2.2
# other pinned dependencies
```

---

## Persistence and backends

Chroma supports different persistence strategies depending on your installation and version: in-memory (default for quickstarts) and on-disk persistence for durability.

- In-memory: fastest for prototyping but data lost on restart.
- On-disk: recommended for production or long-running workloads.
- External stores: sometimes Chroma can be configured with external storage layers or checkpointing — consult specific release docs.

Table: persistence trade-offs

| Mode | Durability | Latency | Use case |
|---|---:|---:|---|
| In-memory | Volatile | Lowest | Prototyping, dev |
| On-disk | Durable | Low–medium | Production single-node |
| External (S3/DB) | Durable (depends) | Medium | Large datasets / backups |

---

## Server & container deployment

- Containerize the application and pin versions.
- Expose metrics and health endpoints for orchestration.
- Use persistent volumes for on-disk storage in container orchestration platforms (Kubernetes PVs, Docker volumes).

Kubernetes notes:

- Deploy Chroma as part of your application pod or as a sidecar depending on architecture.
- Use a PersistentVolumeClaim for the data directory when using on-disk persistence.

---

## Basic Python quickstart (runnable)

Create `quickstart.py`:

```py
from chromadb import Client
client = Client()
col = client.create_collection('quickstart')
col.add(ids=['id1'], embeddings=[[0.1,0.2,0.3]], metadatas=[{'source':'demo'}], documents=['Hello world'])
res = col.query(query_embeddings=[[0.1,0.2,0.3]], n_results=1)
print(res)
```

Run:

```bash
python quickstart.py
```

---

## Troubleshooting common install problems

- Permission errors: use virtualenv and avoid installing into system Python.
- Incompatible Python versions: ensure Python >=3.9.
- Network/timeouts when downloading packages: set `HTTP_PROXY`/`HTTPS_PROXY` if behind a proxy.
- Missing optional features: check the package extras and install accordingly.

---

## Security and credentials

- Never commit API keys or credentials to source control.
- Use environment variables or secrets management (Kubernetes Secrets, HashiCorp Vault) in production.

