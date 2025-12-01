# Quick Start Guide

## Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- OpenAI API Key (or Azure OpenAI credentials)

## Setup

### 1. Clone and Configure

```bash
# Copy environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-key-here
```

### 2. Start Services with Docker Compose

```bash
docker-compose up -d
```

This will start:
- PostgreSQL (port 5432)
- MongoDB (port 27017)
- Redis (port 6379)
- Kafka (port 9092)
- Backend API (port 8000)
- Frontend (port 3000)

### 3. Initialize Database

```bash
# Run database initialization script
cd backend
python scripts/init_db.py
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Testing the System

### Create a Ticket via API

```bash
curl -X POST http://localhost:8000/api/v1/tickets/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Server Down",
    "description": "Production server is not responding",
    "channel": "web",
    "impact": "high"
  }'
```

### Create a Ticket via Chatbot

1. Navigate to http://localhost:3000/chatbot
2. Type: "My server is down and I can't access my application"
3. The AI agents will automatically:
   - Create the ticket
   - Classify the issue
   - Assign priority and SLA
   - Assign an engineer
   - Attempt auto-resolution

## Development

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

## Production Deployment

### Kubernetes

```bash
# Apply Kubernetes configurations
kubectl apply -f infrastructure/k8s/

# Set secrets
kubectl create secret generic helpdesk-secrets \
  --from-literal=openai-api-key=your-key \
  --from-literal=postgres-password=your-password
```

### Terraform

```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

## Monitoring

Access Prometheus and Grafana:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (if configured)

## Troubleshooting

### Database Connection Issues

Check if services are running:
```bash
docker-compose ps
```

### Kafka Issues

Wait for Kafka to be ready:
```bash
docker-compose logs kafka
```

### API Errors

Check backend logs:
```bash
docker-compose logs backend
```

## Next Steps

1. Configure Azure OpenAI (if using Azure)
2. Add more knowledge base entries
3. Configure monitoring alerts
4. Set up CI/CD pipeline
5. Add more engineers and skills


