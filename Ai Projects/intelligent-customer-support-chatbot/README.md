# Intelligent Customer Support Chatbot

A comprehensive AI-powered customer support chatbot built with FastAPI, OpenAI, and vector databases.

## Features

- 🤖 **AI-Powered Responses**: Uses GPT-4 for intelligent, context-aware responses
- 🧠 **Intent Classification**: Automatically identifies user intent
- 📚 **Knowledge Base**: Vector-based semantic search for FAQ and documentation
- 💬 **Conversation Management**: Multi-turn conversation tracking with context
- 😊 **Sentiment Analysis**: Detects customer sentiment and escalates when needed
- 🔄 **Human Handoff**: Seamless escalation to human agents
- 📊 **Analytics Ready**: Built-in conversation logging and analytics

## Project Structure

```
intelligent-customer-support-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic models
│   ├── database.py             # Database setup
│   ├── nlp_engine.py           # NLP processing
│   ├── knowledge_base.py       # Vector database management
│   ├── response_generator.py   # LLM response generation
│   └── conversation_manager.py # Conversation handling
├── frontend/
│   └── widget.html             # Chat widget UI
├── config.py                   # Configuration
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                   # This file
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key and database credentials
```

5. **Set up database**
```bash
# Make sure PostgreSQL is running
# Update DATABASE_URL in .env
# The database tables will be created automatically on first run
```

6. **Run the application**
```bash
python -m app.main
# Or using uvicorn directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

Edit `.env` file with your settings:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string (optional, for caching)
- `OPENAI_MODEL`: Model to use (default: gpt-4-turbo-preview)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-large)

## API Endpoints

### Create Conversation
```bash
POST /api/v1/conversations
```

### Send Message
```bash
POST /api/v1/conversations/{conversation_id}/messages
Body: {
  "message": "How do I track my order?",
  "conversation_id": "optional-conversation-id"
}
```

### Get Conversation History
```bash
GET /api/v1/conversations/{conversation_id}
```

### Search Knowledge Base
```bash
POST /api/v1/knowledge/search
Body: {
  "query": "return policy",
  "category": "optional-category",
  "limit": 5
}
```

### Add Knowledge Article
```bash
POST /api/v1/knowledge/articles?title=...&content=...&category=...
```

## Frontend Widget

Open `frontend/widget.html` in a browser or integrate it into your website. The widget connects to the API at `http://localhost:8000` by default.

To integrate into your website:
1. Copy the HTML, CSS, and JavaScript from `widget.html`
2. Update the `API_BASE_URL` to point to your deployed API
3. Include the code in your website

## Usage Example

```python
import requests

# Create conversation
response = requests.post("http://localhost:8000/api/v1/conversations")
conversation = response.json()
conversation_id = conversation["conversation_id"]

# Send message
response = requests.post(
    f"http://localhost:8000/api/v1/conversations/{conversation_id}/messages",
    json={"message": "How do I track my order?"}
)
result = response.json()
print(result["response"])
```

## Testing

```bash
pytest
```

## Production Deployment

1. Set `DEBUG=False` in `.env`
2. Use a production-grade ASGI server (e.g., Gunicorn with Uvicorn workers)
3. Set up proper database backups
4. Configure CORS appropriately
5. Use environment variables for all secrets
6. Set up monitoring and logging
7. Use a reverse proxy (nginx) for SSL termination

## License

MIT License

