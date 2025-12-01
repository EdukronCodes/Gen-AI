# 📁 Project Structure

Complete overview of the Agentic AI Social Media Automation System architecture.

## Directory Tree

```
Project 3/
│
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   │
│   │   ├── agents/                   # 🤖 AI Agents
│   │   │   ├── __init__.py
│   │   │   ├── base_agent.py         # Base agent class
│   │   │   ├── strategy_agent.py     # 🎯 Strategy planning
│   │   │   ├── content_writer_agent.py  # ✍️ Content generation
│   │   │   ├── creative_agent.py     # 🎨 Visual content ideas
│   │   │   ├── scheduler_agent.py    # ⏰ Post scheduling
│   │   │   ├── posting_agent.py      # 🤖 Platform posting
│   │   │   ├── analytics_agent.py    # 📊 Performance analysis
│   │   │   ├── optimization_agent.py # 🔁 Optimization recommendations
│   │   │   └── agent_orchestrator.py # 🎭 Agent coordination
│   │   │
│   │   ├── api/                      # 🌐 API Endpoints
│   │   │   └── v1/
│   │   │       ├── router.py         # Main router
│   │   │       └── endpoints/
│   │   │           ├── campaigns.py  # Campaign CRUD
│   │   │           ├── platforms.py  # Platform connections
│   │   │           ├── analytics.py  # Analytics endpoints
│   │   │           └── posts.py      # Post management
│   │   │
│   │   ├── core/                     # ⚙️ Core Configuration
│   │   │   ├── config.py             # Settings & env vars
│   │   │   └── database.py           # DB connections
│   │   │
│   │   ├── models/                   # 💾 Database Models
│   │   │   ├── user.py               # User model
│   │   │   ├── campaign.py           # Campaign & CampaignPost
│   │   │   ├── platform.py           # PlatformConnection & Post
│   │   │   └── analytics.py          # Analytics model
│   │   │
│   │   ├── services/                 # 🔧 Business Logic
│   │   │   ├── campaign_service.py   # Campaign management
│   │   │   └── platform_services.py  # Platform API integrations
│   │   │
│   │   └── utils/                    # 🛠️ Utilities
│   │       └── gemini_client.py      # Gemini API client
│   │
│   ├── celery_app/                   # ⏳ Celery Tasks
│   │   ├── __init__.py
│   │   ├── celery.py                 # Celery app config
│   │   └── tasks.py                  # Scheduled tasks
│   │
│   ├── alembic.ini                   # Database migrations
│   ├── requirements.txt              # Python dependencies
│   ├── .env.example                  # Environment template
│   └── run.py                        # Development server
│
├── frontend/                         # Next.js Frontend
│   ├── app/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx         # 📊 Main dashboard
│   │   │   ├── CampaignForm.tsx      # ➕ Campaign creation
│   │   │   ├── PlatformConnections.tsx  # 🔌 Platform setup
│   │   │   └── Analytics.tsx         # 📈 Analytics view
│   │   ├── layout.tsx                # App layout
│   │   ├── page.tsx                  # Home page
│   │   └── globals.css               # Global styles
│   │
│   ├── package.json                  # Node dependencies
│   ├── next.config.js                # Next.js config
│   ├── tailwind.config.js            # Tailwind CSS config
│   └── .env.example                  # Environment template
│
├── README.md                         # Main documentation
├── SETUP.md                          # Setup instructions
├── PROJECT_STRUCTURE.md              # This file
└── .gitignore                        # Git ignore rules
```

## Architecture Overview

### 🎯 Agent System Flow

```
User Goal
    ↓
Strategy Agent (Plans content strategy)
    ↓
Content Writer Agent (Generates captions, scripts)
    ↓
Creative Agent (Creates visual ideas)
    ↓
Scheduler Agent (Optimizes posting times)
    ↓
Posting Agent (Publishes to platforms)
    ↓
Analytics Agent (Tracks performance)
    ↓
Optimization Agent (Improves future posts)
```

### 🔄 Data Flow

```
Frontend (Next.js)
    ↓ HTTP Requests
FastAPI Backend
    ↓
Agent Orchestrator
    ↓
Individual Agents (Gemini AI)
    ↓
Platform Services (API Calls)
    ↓
Database (PostgreSQL/MongoDB)
```

### 🗄️ Database Schema

**PostgreSQL (Main)**
- `users` - User accounts
- `campaigns` - Campaign definitions
- `campaign_posts` - Generated posts
- `platform_connections` - OAuth tokens
- `posts` - Published posts
- `analytics` - Performance metrics

**MongoDB (Optional)**
- Analytics aggregation
- Historical performance data

### 🔌 API Endpoints

**Campaigns**
- `POST /api/v1/campaigns` - Create campaign
- `GET /api/v1/campaigns` - List campaigns
- `GET /api/v1/campaigns/{id}` - Get campaign
- `POST /api/v1/campaigns/{id}/execute` - Execute posting

**Platforms**
- `POST /api/v1/platforms/connect` - Connect platform
- `GET /api/v1/platforms` - List connections
- `DELETE /api/v1/platforms/{id}` - Disconnect

**Analytics**
- `GET /api/v1/analytics/campaign/{id}` - Campaign analytics
- `GET /api/v1/analytics/post/{id}` - Post analytics
- `GET /api/v1/analytics/optimize/{id}` - Optimizations

**Posts**
- `POST /api/v1/posts` - Create/post content
- `GET /api/v1/posts` - List posts

## Technology Stack

### Backend
- **FastAPI** - Web framework
- **SQLAlchemy** - ORM
- **PostgreSQL** - Primary database
- **MongoDB** - Analytics storage
- **Celery** - Task queue
- **Redis** - Message broker
- **Google Gemini** - LLM

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Axios** - HTTP client

## Key Features by Component

### Agents
1. **Strategy Agent** - Analyzes goals, creates content plan
2. **Content Writer** - Generates captions, threads, scripts
3. **Creative Agent** - Visual content ideas and prompts
4. **Scheduler** - Optimizes posting times
5. **Posting Agent** - Handles API calls to platforms
6. **Analytics Agent** - Performance tracking
7. **Optimization Agent** - Improvement recommendations

### Services
- **Campaign Service** - Campaign lifecycle management
- **Platform Services** - Instagram, Facebook, Twitter, YouTube APIs

### Frontend Components
- **Dashboard** - Overview and stats
- **Campaign Form** - Create campaigns
- **Platform Connections** - Manage API connections
- **Analytics** - Performance visualization

## Extension Points

### Adding New Platforms
1. Create service in `platform_services.py`
2. Add platform to `posting_agent.py`
3. Update frontend platform list

### Adding New Agent Types
1. Create agent class in `agents/`
2. Extend `BaseAgent`
3. Add to orchestrator workflow

### Custom Analytics
1. Extend `AnalyticsAgent`
2. Add new metrics to database
3. Create visualization in frontend

This architecture provides a scalable, maintainable foundation for multi-platform social media automation.

