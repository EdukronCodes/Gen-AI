# ⚡ Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] Node.js 18+ installed
- [ ] PostgreSQL installed and running
- [ ] Redis installed and running (optional for now)
- [ ] Google Gemini API key

## 1. Backend Setup (2 minutes)

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env  # Windows
# OR
cp .env.example .env    # Mac/Linux

# Edit .env and add your GEMINI_API_KEY
# GEMINI_API_KEY=your_key_here

# Start backend
python run.py
```

✅ Backend running at `http://localhost:8000`

## 2. Frontend Setup (2 minutes)

```bash
# Open new terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env.local
copy .env.example .env.local  # Windows
# OR
cp .env.example .env.local    # Mac/Linux

# Start frontend
npm run dev
```

✅ Frontend running at `http://localhost:3000`

## 3. Test It Out (1 minute)

1. Open browser to `http://localhost:3000`
2. Click **Campaigns** tab
3. Fill in the form:
   - Name: "Test Campaign"
   - Goal: "Promote my product for 7 days"
   - Select platforms: Instagram, Twitter
   - Duration: 7 days
4. Click **Create Campaign & Generate Strategy**

🎉 Watch the AI agents work! The Strategy Agent will create a content plan.

## What Happens Next?

1. **Strategy Agent** analyzes your goal
2. **Content Writer Agent** generates captions
3. **Creative Agent** creates visual ideas
4. **Scheduler Agent** optimizes posting times
5. Posts are created and stored in database

## View Your Campaign

- Go to **Dashboard** to see your campaigns
- Check **Analytics** to see performance (after posts are published)

## Need Help?

- API Docs: `http://localhost:8000/docs`
- Backend logs: Check terminal where backend is running
- Frontend errors: Check browser console (F12)

## Next Steps

1. Connect social media platforms (see SETUP.md)
2. Execute campaigns to actually post
3. Monitor analytics
4. Get optimization recommendations

🚀 You're ready to automate your social media!

