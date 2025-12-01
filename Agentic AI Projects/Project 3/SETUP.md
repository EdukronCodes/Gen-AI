# 🚀 Setup Guide

Complete setup instructions for the Agentic AI Social Media Automation System.

## Prerequisites

### Backend Requirements
- Python 3.11 or higher
- PostgreSQL 12+
- Redis 6+
- MongoDB 4.4+ (optional, for analytics)

### Frontend Requirements
- Node.js 18+ and npm

### API Keys Required
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))
- Social Media API credentials (Instagram, Facebook, Twitter, YouTube)

## Step-by-Step Setup

### 1. Clone and Navigate

```bash
cd "Project 3"
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure Environment

Create `.env` file in `backend/` directory:

```bash
cp .env.example .env
```

Edit `.env` with your configurations:

```env
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/social_automation
MONGODB_URL=mongodb://localhost:27017/social_automation

# Redis
REDIS_URL=redis://localhost:6379/0

# AI/LLM - REQUIRED
GEMINI_API_KEY=your_gemini_api_key_here

# Social Media APIs (add when ready)
INSTAGRAM_ACCESS_TOKEN=
FACEBOOK_ACCESS_TOKEN=
TWITTER_API_KEY=
TWITTER_API_SECRET=
TWITTER_BEARER_TOKEN=
YOUTUBE_CLIENT_ID=
YOUTUBE_CLIENT_SECRET=

# Security
SECRET_KEY=your-secret-key-change-this
```

#### Initialize Database

```bash
# Create database first
createdb social_automation  # PostgreSQL command

# Run migrations (when Alembic is configured)
# alembic upgrade head
```

#### Start Backend Server

```bash
python run.py
# OR
uvicorn app.main:app --reload
```

Backend will be available at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

### 3. Frontend Setup

#### Install Dependencies

```bash
cd ../frontend
npm install
```

#### Configure Environment

Create `.env.local` file:

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### Start Frontend

```bash
npm run dev
```

Frontend will be available at `http://localhost:3000`

### 4. Celery Worker (Optional - for Scheduling)

In a separate terminal:

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
celery -A celery_app.celery worker --loglevel=info
```

For scheduled tasks, also run Celery Beat:

```bash
celery -A celery_app.celery beat --loglevel=info
```

## 5. Getting API Keys

### Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Create API key
4. Copy and paste into `.env` file

### Social Media APIs

#### Instagram
1. Go to [Meta for Developers](https://developers.facebook.com/)
2. Create an app
3. Add Instagram Basic Display or Instagram Graph API
4. Generate access tokens

#### Facebook
1. Use Meta for Developers
2. Create app with Facebook Login
3. Generate Page Access Token

#### Twitter/X
1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Create a project and app
3. Generate API keys and bearer token

#### YouTube
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable YouTube Data API v3
3. Create OAuth 2.0 credentials
4. Get client ID and secret

## Quick Start Test

1. **Start Backend**: `cd backend && python run.py`
2. **Start Frontend**: `cd frontend && npm run dev`
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Create Campaign**: 
   - Go to Campaigns tab
   - Fill in the form
   - Click "Create Campaign & Generate Strategy"
   - Watch AI agents work!

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Change port in run.py or use:
uvicorn app.main:app --port 8001
```

**Database connection error:**
- Check PostgreSQL is running
- Verify DATABASE_URL in .env
- Ensure database exists

**Gemini API error:**
- Verify GEMINI_API_KEY is set correctly
- Check API quota limits

### Frontend Issues

**Cannot connect to API:**
- Check NEXT_PUBLIC_API_URL in .env.local
- Ensure backend is running
- Check CORS settings in backend

**Build errors:**
```bash
rm -rf node_modules .next
npm install
npm run dev
```

## Project Structure

```
Project 3/
├── backend/
│   ├── app/
│   │   ├── agents/          # AI agents
│   │   ├── api/             # API endpoints
│   │   ├── core/            # Config & database
│   │   ├── models/          # Database models
│   │   ├── services/        # Business logic
│   │   └── utils/           # Utilities
│   ├── celery_app/          # Celery tasks
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── app/
│   │   ├── components/      # React components
│   │   └── page.tsx         # Main page
│   ├── package.json
│   └── .env.local
└── README.md
```

## Next Steps

1. ✅ Set up database and run migrations
2. ✅ Get Gemini API key
3. ✅ Test campaign creation
4. ✅ Connect social media platforms
5. ✅ Schedule and post content
6. ✅ Monitor analytics

## Support

For issues or questions:
- Check API documentation at `/docs` endpoint
- Review agent logs in console
- Check database for stored data

Happy automating! 🚀

