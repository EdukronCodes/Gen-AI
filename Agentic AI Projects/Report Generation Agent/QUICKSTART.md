# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Environment

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

## Step 3: Create Database (if not already created)

```bash
python create_database.py
```

This will create `retail_banking.db` with 20 tables and sample data.

## Step 4: Start the Server

```bash
python start_server.py
```

Or manually:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## Step 5: Use the Frontend

1. Open `frontend/index.html` in your web browser
2. Enter a question about the database
3. Click "Ask Question & Generate PDF"
4. Wait for processing (Query → Analysis → PDF generation)
5. Download the generated PDF report

## Step 6: Test the Framework (Optional)

```bash
python test_framework.py
```

## Example Questions to Try

- "What are the top 5 customers by total account balance?"
- "Show me all active loans with their payment history"
- "What products have the highest sales?"
- "Which branch has the most employees?"
- "Show me customers who have both checking and savings accounts"

## API Usage

### Using curl:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 5 customers by total account balance?"}'
```

### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What are the top 5 customers by total account balance?"}
)

result = response.json()
print(result)
```

## Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure you created `.env` file with your API key
- Check that the key starts with `sk-`

### "Database not found"
- Run `python create_database.py` to create the database

### "Module not found"
- Install dependencies: `pip install -r requirements.txt`

### Frontend can't connect to backend
- Make sure the server is running on `http://localhost:8000`
- Check browser console for CORS errors
- Verify the API_BASE_URL in `frontend/index.html`

## Architecture Overview

```
User Question (Frontend)
    ↓
FastAPI Backend (/ask endpoint)
    ↓
Orchestrator Agent
    ↓
┌─────────────────────────────────┐
│ 1. Query Agent                  │
│    - Converts NL to SQL         │
│    - Executes database query    │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 2. Analysis Agent               │
│    - Analyzes results           │
│    - Generates insights (OpenAI)│
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 3. PDF Agent                    │
│    - Creates formatted PDF      │
│    - Includes summary & data    │
└─────────────────────────────────┘
    ↓
PDF Report (Downloadable)
```

## Next Steps

- Customize agents in `agents/` directory
- Add more database tables
- Enhance PDF templates
- Add authentication
- Deploy to production

