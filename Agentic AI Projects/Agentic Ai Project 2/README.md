# Multi-Agent Airlines System

An intelligent multi-agent system for airline operations built with LlamaIndex, FastAPI, and PostgreSQL. This system features 7 specialized AI agents that handle different aspects of airline services, all orchestrated through an intelligent routing system.

## 🎯 Features

- **7 Specialized Agents:**
  1. **Flight Search Agent** - Finds and searches available flights
  2. **Flight Booking Agent** - Creates and manages reservations
  3. **Customer Service Agent** - Handles general inquiries and support
  4. **Baggage Information Agent** - Provides baggage policies and tracking
  5. **Check-in Agent** - Processes flight check-ins
  6. **Flight Status Agent** - Provides real-time flight status
  7. **Rewards & Loyalty Agent** - Manages loyalty programs and points

- **LlamaIndex Integration** - RAG (Retrieval-Augmented Generation) for domain-specific knowledge
- **PostgreSQL Database** - Comprehensive airline database with flights, bookings, passengers, and more
- **Azure Deployment Ready** - Docker containerization and Azure configuration files
- **FastAPI REST API** - Modern, fast API with automatic documentation

## 🏗️ Architecture

```
┌─────────────────┐
│   FastAPI App   │
└────────┬────────┘
         │
┌────────▼────────┐
│  Orchestrator   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐  ┌──▼───┐  ... (7 Agents)
│ Agent │  │Agent │
└───┬───┘  └──┬───┘
    │         │
┌───▼─────────▼───┐
│  PostgreSQL DB  │
└─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 12+ (or Azure Database for PostgreSQL)
- OpenAI API Key
- Docker (for containerization)
- Azure account (for deployment)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd "Agentic Ai Project 2"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your credentials
# - OPENAI_API_KEY
# - DATABASE_URL
```

### 3. Initialize Database

```bash
# Run database initialization script
python scripts/init_database.py
```

This will:
- Create all database tables
- Seed airlines, airports, flights, passengers, and bookings

### 4. Run the Application

```bash
# Start the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

## 📡 API Endpoints

### Query Endpoint
```bash
POST /api/query
{
  "query": "Find flights from New York to London",
  "context": {}
}
```

### Search Flights
```bash
GET /api/flights?departure_city=New York&arrival_city=London
```

### Create Booking
```bash
POST /api/book
{
  "passenger_email": "john.doe@email.com",
  "flight_id": 1,
  "seat_class": "economy",
  "special_requests": "Window seat preferred"
}
```

### Check-in
```bash
POST /api/checkin
{
  "booking_reference": "ABC123",
  "seat_number": "12A"
}
```

### Get Agents Info
```bash
GET /api/agents
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t airlines-multi-agent:latest .
```

### Run Container
```bash
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  -e OPENAI_API_KEY="your-key" \
  airlines-multi-agent:latest
```

## ☁️ Azure Deployment

### Option 1: Azure Container Instances (ACI)

1. **Build and push to Azure Container Registry:**
```bash
# Login to Azure
az login

# Create resource group
az group create --name airlines-rg --location eastus

# Create container registry
az acr create --resource-group airlines-rg --name yourregistry --sku Basic

# Build and push image
az acr build --registry yourregistry --image airlines-system:latest .
```

2. **Deploy to ACI:**
```bash
az container create \
  --resource-group airlines-rg \
  --name airlines-multi-agent \
  --image yourregistry.azurecr.io/airlines-system:latest \
  --dns-name-label airlines-multi-agent \
  --ports 8000 \
  --environment-variables \
    DATABASE_URL="your-azure-postgres-connection-string" \
    OPENAI_API_KEY="your-openai-key" \
  --cpu 2 \
  --memory 4
```

### Option 2: Azure App Service

1. **Create App Service:**
```bash
az webapp create \
  --resource-group airlines-rg \
  --plan airlines-plan \
  --name airlines-multi-agent \
  --deployment-container-image-name yourregistry.azurecr.io/airlines-system:latest
```

2. **Configure environment variables:**
```bash
az webapp config appsettings set \
  --resource-group airlines-rg \
  --name airlines-multi-agent \
  --settings \
    DATABASE_URL="your-connection-string" \
    OPENAI_API_KEY="your-key"
```

### Option 3: Azure Kubernetes Service (AKS)

See `azure-pipelines.yml` for CI/CD pipeline configuration.

## 🗄️ Database Schema

The system includes the following main tables:

- **airlines** - Airline companies
- **airports** - Airport information
- **flights** - Flight schedules and details
- **passengers** - Passenger information
- **bookings** - Flight reservations
- **baggage** - Baggage tracking
- **flight_status_updates** - Real-time flight status

## 🤖 Agents Overview

### Flight Search Agent
- Searches flights by route, date, airline
- Filters by price and availability
- Returns detailed flight information

### Flight Booking Agent
- Creates new reservations
- Manages booking references
- Handles seat class selection

### Customer Service Agent
- General inquiries and support
- Booking modifications
- Cancellation and refunds

### Baggage Agent
- Baggage allowance information
- Tracking by tracking number
- Policy and restriction details

### Check-in Agent
- Online check-in processing
- Seat assignment
- Boarding pass generation

### Flight Status Agent
- Real-time flight status
- Delay information
- Gate and terminal details

### Rewards Agent
- Loyalty program information
- Points balance and redemption
- Membership tier benefits

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY` - Required for LLM functionality
- `DATABASE_URL` - PostgreSQL connection string
- `AZURE_TENANT_ID` - Optional, for Azure Key Vault
- `AZURE_CLIENT_ID` - Optional, for Azure Key Vault
- `AZURE_CLIENT_SECRET` - Optional, for Azure Key Vault

### Database Connection

For Azure PostgreSQL, use connection string format:
```
postgresql://username:password@server.postgres.database.azure.com:5432/dbname?sslmode=require
```

## 📝 Example Usage

### Python Client Example
```python
import requests

# Query the system
response = requests.post("http://localhost:8000/api/query", json={
    "query": "I want to book a flight from New York to London",
    "context": {
        "passenger_email": "john.doe@email.com"
    }
})

print(response.json())
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is my baggage allowance?",
    "context": {
      "booking_reference": "ABC123"
    }
  }'
```

## 🧪 Testing

```bash
# Test health endpoint
curl http://localhost:8000/api/health

# List all agents
curl http://localhost:8000/api/agents

# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find flights from JFK to LHR"}'
```

## 📦 Project Structure

```
.
├── agents/                 # Agent implementations
│   ├── base_agent.py
│   ├── flight_search_agent.py
│   ├── flight_booking_agent.py
│   ├── customer_service_agent.py
│   ├── baggage_agent.py
│   ├── checkin_agent.py
│   ├── flight_status_agent.py
│   └── rewards_agent.py
├── database/              # Database models and setup
│   ├── models.py
│   ├── database.py
│   └── seed_data.py
├── orchestrator/          # Agent orchestration
│   └── agent_orchestrator.py
├── scripts/               # Utility scripts
│   └── init_database.py
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── azure-deploy.yml      # Azure deployment config
└── README.md            # This file
```

## 🔒 Security Notes

- Never commit `.env` file with real credentials
- Use Azure Key Vault for production secrets
- Enable SSL/TLS for database connections
- Implement authentication for production deployment
- Use environment variables for sensitive data

## 🐛 Troubleshooting

### Database Connection Issues
- Verify `DATABASE_URL` is correct
- Check PostgreSQL is running and accessible
- For Azure, ensure firewall rules allow your IP

### OpenAI API Issues
- Verify `OPENAI_API_KEY` is set correctly
- Check API quota and billing
- Ensure network connectivity

### Agent Routing Issues
- Check logs for routing decisions
- Verify agent initialization
- Test individual agents directly

## 📄 License

This project is provided as-is for demonstration purposes.

## 🤝 Contributing

This is a demonstration project. For production use, consider:
- Adding authentication and authorization
- Implementing rate limiting
- Adding comprehensive error handling
- Creating unit and integration tests
- Setting up monitoring and logging

## 📞 Support

For issues or questions, please check:
- API documentation at `/docs` endpoint
- Agent information at `/api/agents` endpoint
- Health status at `/api/health` endpoint


