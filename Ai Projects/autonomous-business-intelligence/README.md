# Autonomous Business Intelligence Using Agentic AI

A comprehensive autonomous business intelligence system powered by agentic AI that independently discovers insights, generates reports, and provides actionable recommendations.

## Features

- 🤖 **Multi-Agent System**: Specialized AI agents working collaboratively
- 🧠 **Autonomous Operation**: Self-directed analysis and insight generation
- 📊 **Predictive Analytics**: Time series forecasting and trend prediction
- 🔍 **Anomaly Detection**: Automatic detection of outliers and anomalies
- 💡 **Insight Generation**: AI-powered business insights with recommendations
- 📈 **Natural Language Queries**: Query data using conversational language
- 📋 **Automated Reporting**: Self-generating reports
- 🔄 **Real-Time Monitoring**: Continuous KPI tracking

## Project Structure

```
autonomous-business-intelligence/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── models.py                  # Pydantic models
│   ├── database.py                # Database setup
│   ├── insight_generator.py       # Insight generation
│   └── agents/
│       ├── base_agent.py          # Base agent class
│       ├── orchestrator_agent.py  # Master orchestrator
│       ├── data_collection_agent.py
│       ├── analysis_agent.py
│       ├── anomaly_detection_agent.py
│       └── predictive_agent.py
├── config.py                      # Configuration
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                      # This file
```

## Installation

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key and database credentials
```

4. **Set up databases**
```bash
# PostgreSQL for structured data
# MongoDB for analytics data
# Redis for caching
# Update connection strings in .env
```

5. **Run the application**
```bash
python -m app.main
# Or using uvicorn:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## Configuration

Edit `.env` file with your settings:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DATABASE_URL`: PostgreSQL connection string
- `MONGODB_URL`: MongoDB connection string
- `REDIS_URL`: Redis connection string
- `OPENAI_MODEL`: Model to use (default: gpt-4-turbo-preview)

## API Endpoints

### Process Natural Language Query
```bash
POST /api/v1/query
Body: {
  "query": "What caused the sales drop in Q3?",
  "format": "json"
}
```

### Process Query Asynchronously
```bash
POST /api/v1/query/async
Body: {
  "query": "Show me revenue trends for the last 6 months"
}
```

### Get Query Result
```bash
GET /api/v1/query/{task_id}
```

### List Agents
```bash
GET /api/v1/agents
```

### Get Agent Status
```bash
GET /api/v1/agents/{agent_id}/status
```

### Assign Task to Agent
```bash
POST /api/v1/agents/{agent_id}/task
Body: {
  "task_type": "analysis",
  "parameters": {...}
}
```

### Get Insights
```bash
GET /api/v1/insights?category=sales&priority=high
```

### Generate Report
```bash
POST /api/v1/reports/generate
Body: {
  "name": "Monthly Sales Report",
  "template": "sales_template",
  "format": "pdf"
}
```

### List Data Sources
```bash
GET /api/v1/data-sources
```

### Add Data Source
```bash
POST /api/v1/data-sources
Body: {
  "name": "Sales Database",
  "type": "database",
  "connection": {...}
}
```

## Agent System

### Orchestrator Agent
Coordinates all agents and manages task decomposition and result synthesis.

### Specialized Agents

1. **Data Collection Agent**: Collects data from databases, APIs, and files
2. **Analysis Agent**: Performs statistical analysis, correlation analysis, trend analysis
3. **Anomaly Detection Agent**: Detects anomalies using Isolation Forest, statistical methods, Z-score
4. **Predictive Agent**: Performs time series forecasting and trend prediction

## Usage Example

```python
import requests

# Process natural language query
response = requests.post(
    "http://localhost:8001/api/v1/query",
    json={
        "query": "What are the sales trends for the last quarter?",
        "format": "json"
    }
)
result = response.json()
print(result["response"])
print(result["insights"])

# Get insights
response = requests.get("http://localhost:8001/api/v1/insights")
insights = response.json()
for insight in insights:
    print(f"{insight['title']}: {insight['description']}")
    print(f"Recommendations: {insight['recommendations']}")
```

## Agent Capabilities

### Data Collection Agent
- Database queries (SQL)
- API data fetching
- File reading (CSV, Excel, JSON)
- Data validation

### Analysis Agent
- Statistical analysis
- Correlation analysis
- Trend analysis
- Cohort analysis

### Anomaly Detection Agent
- Isolation Forest detection
- Statistical anomaly detection
- Z-score based detection

### Predictive Agent
- Time series forecasting
- Linear trend forecasting
- Demand forecasting

## Testing

```bash
pytest
```

## Production Deployment

1. Set `DEBUG=False` in `.env`
2. Use production-grade ASGI server
3. Set up proper database backups
4. Configure CORS appropriately
5. Use environment variables for all secrets
6. Set up monitoring and logging
7. Use a reverse proxy (nginx) for SSL termination
8. Set up task queue (Celery) for background processing

## License

MIT License

