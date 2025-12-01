# Architecture Documentation

## System Architecture

### Overview
The Agentic AI IT Help Desk system is built on a microservices architecture with multiple AI agents collaborating to autonomously handle IT incidents.

### Components

#### 1. Agent Layer
- **Orchestrator Agent**: Coordinates all agents
- **Intake Agent**: Handles ticket creation from multiple channels
- **Classification Agent**: Classifies issues using LLM + RAG
- **SLA Agent**: Assigns priority and tracks SLA
- **Assignment Agent**: Auto-assigns engineers
- **Resolution Agent**: Executes auto-fix scripts
- **Monitoring Agent**: Monitors infrastructure
- **Escalation Agent**: Manages escalations
- **RCA Agent**: Generates root cause analysis
- **Reporting Agent**: Creates management reports

#### 2. Backend Services
- FastAPI REST API
- WebSocket for real-time updates
- PostgreSQL for transactional data
- MongoDB for logs
- Redis for caching
- Kafka for event streaming

#### 3. Frontend
- React.js SPA
- Real-time dashboard
- Chatbot interface
- Ticket management UI

#### 4. Infrastructure
- Docker containers
- Kubernetes orchestration
- CI/CD pipelines
- Monitoring & observability

## Data Flow

1. User creates ticket → Intake Agent
2. Intake Agent → Classification Agent
3. Classification Agent → SLA Agent
4. SLA Agent → Assignment Agent
5. Assignment Agent → Resolution Agent
6. Resolution Agent → Monitoring/Escalation Agents
7. All agents → RCA Agent
8. RCA Agent → Reporting Agent

## Security

- OAuth2 + JWT authentication
- RBAC for role-based access
- Encrypted vector database
- Secrets in Azure Key Vault
- AI request auditing


