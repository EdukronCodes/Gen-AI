# 🚀 Launch Instructions

## Quick Start

### Option 1: Using the Batch File (Windows)
```bash
start_server.bat
```

### Option 2: Using Python Script
```bash
python run.py
```

### Option 3: Direct Uvicorn
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

## Access Points

Once the server starts, open these URLs in your browser:

### 🌐 Web UI (Main Interface)
```
http://localhost:8001/ui
```

### 📡 API Documentation
```
http://localhost:8001/docs
```

### 💚 Health Check
```
http://localhost:8001/api/health
```

### 🤖 Agents List
```
http://localhost:8001/api/agents
```

## Troubleshooting

### Port Already in Use
If port 8001 is busy, change it:
```bash
set PORT=8002
python run.py
```

### Database Issues
The system uses SQLite by default (no setup needed).
If you want PostgreSQL, set DATABASE_URL in .env file.

### Missing OpenAI API Key
The system works without it, but LLM features require:
```
OPENAI_API_KEY=your_key_here
```

## Features Available

✅ 7 Specialized AI Agents
✅ Real-time Chat Interface  
✅ Flight Search & Booking
✅ Check-in Services
✅ Baggage Information
✅ Flight Status Updates
✅ Loyalty Program Management
✅ Beautiful Modern UI

Enjoy! ✈️


