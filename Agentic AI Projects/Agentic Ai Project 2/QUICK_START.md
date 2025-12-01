# 🚀 Quick Start Guide

## Access the Application

Once the server is running, you can access:

### 🌐 Web UI (Recommended)
**Open in your browser:**
```
http://localhost:8000/ui
```

### 📡 API Endpoints

**API Documentation (Swagger):**
```
http://localhost:8000/docs
```

**Alternative API Docs (ReDoc):**
```
http://localhost:8000/redoc
```

**Health Check:**
```
http://localhost:8000/api/health
```

**List All Agents:**
```
http://localhost:8000/api/agents
```

## 🎯 Try These Queries in the UI

1. **Search for flights:**
   - "Find flights from New York to London"
   - "Search flights from JFK to LHR"

2. **Check flight status:**
   - "What's the status of flight AA100?"
   - "Check flight status"

3. **Baggage information:**
   - "What is my baggage allowance?"
   - "Tell me about baggage policies"

4. **Loyalty program:**
   - "How many points do I have?"
   - "Tell me about my membership tier"

5. **General inquiries:**
   - "Help me with my booking"
   - "What are your cancellation policies?"

## 🔧 If Server is Not Running

Start it manually:
```bash
python run.py
```

Or directly:
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📝 Environment Setup

Make sure you have a `.env` file with:
```
OPENAI_API_KEY=your_key_here
DATABASE_URL=sqlite:///./airlines.db
```

The system will work without OpenAI API key for basic functionality, but LLM features require it.

## 🎨 UI Features

- **Real-time Chat Interface** - Talk to AI agents naturally
- **Agent Visualization** - See which agent handles your query
- **Quick Actions** - One-click common queries
- **System Status** - Monitor agent availability
- **Responsive Design** - Works on desktop and mobile

Enjoy your Multi-Agent Airlines System! ✈️


