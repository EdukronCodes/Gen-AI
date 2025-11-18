# Multi-Agent AI Framework

A sophisticated multi-agent and multi-tool agentic AI framework that allows users to query a SQLite database through natural language questions. The system uses multiple specialized AI agents to process queries, analyze results, and generate comprehensive PDF summaries.

## Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for querying, analysis, and PDF generation
- 🗄️ **Database Integration**: Query SQLite database with natural language
- 📊 **AI-Powered Analysis**: Azure OpenAI GPT-4 for intelligent data analysis
- 📄 **PDF Generation**: Automatic generation of detailed PDF reports
- 🌐 **Web Interface**: Modern, responsive frontend for easy interaction
- 🔄 **Orchestrated Workflow**: Coordinated multi-agent execution

## Architecture

### Agents

1. **QueryAgent**: Converts natural language to SQL and executes database queries
2. **AnalysisAgent**: Analyzes query results and generates insights using Azure OpenAI
3. **PDFAgent**: Creates comprehensive PDF summaries with formatted results
4. **OrchestratorAgent**: Coordinates all agents in a sequential workflow

### Workflow

```
User Question → Orchestrator → QueryAgent → AnalysisAgent → PDFAgent → PDF Report
```

## Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Azure OpenAI credentials:
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

4. **Ensure the database exists**:
```bash
# If you haven't created it yet, run:
python create_database.py
```

## Usage

### Starting the Backend Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Using the Frontend

1. Open `frontend/index.html` in a web browser
2. Enter your question in natural language
3. Click "Ask Question & Generate PDF"
4. Wait for the agents to process your query
5. Download the generated PDF report

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /agents` - Get agent status
- `POST /ask` - Ask a question (returns PDF path and summary)
- `GET /download/{filename}` - Download generated PDF
- `GET /pdfs` - List all available PDFs

### Example API Request

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 5 customers by total account balance?"}'
```

## Example Questions

- "What are the top 5 customers by total account balance?"
- "Show me all active loans with their payment history"
- "What products have the highest sales?"
- "Which branch has the most employees?"
- "Show me customers who have both checking and savings accounts"
- "What is the average transaction amount by account type?"
- "List all products that need restocking (below reorder level)"
- "Show me the most reviewed products with their ratings"

## Project Structure

```
.
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class
│   ├── query_agent.py         # Database query agent
│   ├── analysis_agent.py      # Data analysis agent
│   ├── pdf_agent.py           # PDF generation agent
│   └── orchestrator.py        # Agent orchestrator
├── frontend/
│   └── index.html             # Web interface
├── pdf_outputs/               # Generated PDFs (created automatically)
├── create_database.py         # Database setup script
├── view_relationships.py      # Database verification script
├── main.py                    # FastAPI backend
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Database Schema

The system works with a retail banking database containing 20 related tables:
- Customers, Accounts, Transactions
- Loans, Payments, Credit Cards
- Products, Orders, Inventory
- Employees, Branches, Departments
- And more...

See `create_database.py` for the complete schema.

## Technologies Used

- **FastAPI**: Modern Python web framework
- **Azure OpenAI GPT-4**: AI-powered analysis and SQL generation
- **SQLite**: Database
- **ReportLab**: PDF generation
- **HTML/CSS/JavaScript**: Frontend interface

## Notes

- Make sure you have Azure OpenAI credentials configured in `.env`
- The `AZURE_OPENAI_DEPLOYMENT_NAME` should match your deployment name in Azure (e.g., "gpt-4", "gpt-35-turbo")
- The database file (`retail_banking.db`) must exist in the project root
- PDFs are saved in the `pdf_outputs/` directory
- The frontend expects the backend to be running on `http://localhost:8000`

## License

This project is provided as-is for educational and development purposes.

