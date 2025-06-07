# AI-Powered Retail Chatbot using Gemini API

This project implements a conversational AI chatbot for retail, leveraging the Gemini API (Google's generative AI) to answer customer queries, recommend products, and provide support. The chatbot is accessible via a web interface and can be integrated into e-commerce or retail platforms.

## Features
- Real-time chat interface for customers
- Integration with Gemini API for natural language understanding and generation
- Product recommendation and Q&A
- Chat history (optional, for returning users or analytics)
- Admin interface for monitoring conversations (optional)

## Project Structure

```
ai_retail_chatbot/
├── backend/                 # Django backend (API, Gemini integration)
│   ├── api/                # API endpoints
│   ├── services/           # Gemini API integration logic
│   ├── models/             # (Optional) Store chat logs, user profiles, etc.
│   └── chatbot/            # Django project settings
├── frontend/                # React frontend
│   ├── public/             # Static files
│   └── src/                # Source code
│       ├── components/     # Chat UI, message bubbles, etc.
│       ├── pages/          # Main chat page
│       ├── services/       # API services
│       └── utils/          # Utility functions
└── README.md                # Project documentation
```

## Technology Stack

### Backend
- Python 3.8+
- Django 4.2
- Django REST Framework
- Gemini API (Google Generative AI)
- SQLite (development) / PostgreSQL (production)

### Frontend
- React 18
- Material-UI
- Axios
- React Router

## Setup Instructions

1. Clone the repository
2. Set up the backend:
   ```bash
   cd ai_retail_chatbot/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python manage.py migrate
   python manage.py runserver
   ```

3. Set up the frontend:
   ```bash
   cd ai_retail_chatbot/frontend
   npm install
   npm start
   ```

4. Access the application at http://localhost:3000

## API Endpoints
- `POST /api/chat/` - Send a message to the chatbot and receive a response
- `GET /api/chat/history/` - (Optional) Retrieve chat history

## Environment Variables
- `GEMINI_API_KEY` - Your Gemini API key (set in backend environment)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 