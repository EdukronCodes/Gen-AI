## Agentic AI & Generative AI Document Intelligence Platform (NTT Bank)

### 1. Project Overview
For **NTT Bank**, this project delivers an **Agentic AI–driven document intelligence platform** that processes financial PDFs, validates compliance, protects sensitive data, and generates **business insights dashboards**. The system combines:
- **Agentic AI workflows (LangGraph / AI agents)**
- **RAG-based architecture**
- **Generative AI**

to automate document validation, perform **GDPR-compliant data masking**, and extract actionable insights from financial and sales documents.

### 2. Business Problem & Motivation
- Banks handle large volumes of **financial documents, contracts, statements, and reports**.
- Manual validation for:
  - Regulatory compliance,
  - Sensitive data exposure,
  - Transaction consistency,
is **time-consuming, error-prone, and expensive**.

Key challenges:
- Ensuring **GDPR compliance** and avoiding leakage of personal data.
- Quickly identifying **compliance violations or anomalies**.
- Extracting **sales and transaction insights** from unstructured documents.

The solution aims to **automate compliance checks and insight generation** while keeping data secure.

### 3. High-Level Architecture
- **Document Ingestion Layer**
  - Uploads and processes PDFs and other financial documents.
  - Performs parsing and text extraction, including handling scanned documents.

- **Agentic Workflow (LangGraph)**
  - Orchestrates multiple specialized agents:
    - **Classification Agent** – Identifies document type (statement, contract, invoice, KYC doc, etc.).
    - **Extraction Agent** – Extracts key fields (amounts, dates, parties, account details, personal data).
    - **Compliance Agent (Checker)** – Runs rules and RAG-based checks against regulatory and internal policies.
    - **Masking/Redaction Agent (Maker)** – Performs data masking and regenerates GDPR-compliant document variants.
    - **Insight Agent** – Summarizes transactions, sales performance, and key risk indicators.

- **RAG (Retrieval-Augmented Generation) Layer**
  - Uses **vector search and knowledge retrieval** over:
    - Regulatory documents,
    - Internal compliance policies,
    - Product and process documentation.
  - Supplies contextual snippets to LLMs for **accurate and grounded reasoning**.

- **Dashboard & Analytics Layer**
  - Aggregates outputs to generate:
    - Sales insights,
    - Transaction analysis,
    - Personal data exposure highlights.

### 4. Checker–Maker Validation Framework
- **Checker (Validation & Detection)**
  - Performs **compliance validation**:
    - Checks required clauses and disclosures.
    - Verifies correctness and consistency of transaction information.
  - Detects **personal and sensitive data**:
    - Names, addresses, account numbers, IDs, emails, phone numbers.
  - Uses a mix of:
    - Pattern/rule-based detection.
    - LLM-based semantic detection for less structured mentions.

- **Maker (Masking & Regeneration)**
  - Applies **data masking and anonymization** strategies:
    - Redaction, pseudonymization, or tokenization, depending on policy.
  - Regenerates **secure PDFs or text views**:
    - Only relevant and non-sensitive information is exposed to target users.
  - Ensures that workflows and analytics maintain **GDPR-compliant handling**.

### 5. RAG-Based Contextual Understanding
- Built a **RAG architecture** where:
  - Policy and regulation documents are chunked and indexed.
  - Queries from agents (e.g., “Does this clause satisfy requirement X?”) are resolved by retrieving **relevant policy segments**.
  - LLMs then combine retrieved context with document content to:
    - Decide compliance status.
    - Suggest corrections or missing clauses.

This reduces hallucinations and grounds AI decisions in **actual regulatory text**.

### 6. Dashboards & Insights Generation
- Generated dashboards and reports that show:
  - **Sales performance insights**: revenue trends, product-wise breakdowns.
  - **Transaction analysis**: volumes, anomalies, patterns across clients or regions.
  - **Personal data highlights**: locations and types of sensitive data found.
- Insights are consumed by:
  - Compliance teams.
  - Business and sales stakeholders.
  - Audit and risk management teams.

### 7. GDPR Compliance & Data Protection
- Implemented automated **sensitive data detection and masking** to enforce:
  - Data minimization.
  - Least-privilege access.
  - Controlled views for non-privileged users.
- Ensured that:
  - Raw documents with full personal data are processed securely.
  - Derived artifacts (dashboards, summaries) do not expose unnecessary PII.

### 8. Business Impact
- **Improved document compliance accuracy** by combining rule-based checks and LLM understanding.
- **Reduced manual validation effort**, freeing compliance teams from repetitive checks.
- **Enhanced visibility** into financial transactions and sales performance.
- **Strengthened GDPR compliance** through automated data protection.

### 9. Your Roles & Responsibilities (Expanded)
- **Agentic Workflow Design**
  - Designed **Agentic AI workflows using LangGraph** for end-to-end document processing.
  - Defined roles and responsibilities of Checker and Maker agents.

- **RAG Architecture Implementation**
  - Implemented **RAG-based architecture** for contextual document understanding.
  - Connected regulatory and internal policy sources to the retrieval pipeline.

- **Compliance & Data Protection**
  - Built the **Checker–Maker framework**:
    - Checker → compliance validation, transaction review, personal data detection.
    - Maker → data masking and regenerated secure PDFs.
  - Ensured **GDPR compliance** by automating sensitive data protection steps.

- **Insights & Dashboards**
  - Designed and implemented dashboards for:
    - Sales insights.
    - Transaction analytics.
    - Personal data exposure.
  - Worked with stakeholders to align metrics and visuals with their needs.

### 10. Technologies & Tools
- **AI / Agentic**: Agentic AI, LangGraph.
- **GenAI**: OpenAI / LLM integration.
- **RAG**: Vector retrieval over policy and internal knowledge.
- **Documents**: PDF parsing and processing.
- **Compliance & Security**: Data masking, GDPR-aligned workflows.
- **Analytics**: Dashboards for transaction and sales insights.

