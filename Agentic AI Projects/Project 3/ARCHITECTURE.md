# 🏗️ System Architecture

## Overview

The Agentic AI Social Media Automation System uses a multi-agent architecture where autonomous AI agents collaborate to plan, create, schedule, post, analyze, and optimize social media content.

## Core Principles

1. **Agentic AI** - Agents think, decide, and execute autonomously
2. **Multi-Agent Collaboration** - Agents work together in orchestrated workflows
3. **Platform Agnostic** - Unified interface for multiple social platforms
4. **Data-Driven** - Continuous learning and optimization from performance data

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Dashboard │  │Campaigns │  │Platforms │  │Analytics │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                  Next.js + React + Tailwind                  │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP/REST
┌───────────────────────▼─────────────────────────────────────┐
│                   API Gateway Layer                          │
│                  FastAPI Backend                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Campaigns  │  │ Platforms  │  │ Analytics  │            │
│  │ Endpoints  │  │ Endpoints  │  │ Endpoints  │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Agent Orchestration Layer                       │
│              Agent Orchestrator (LangGraph)                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Agent Workflow Pipeline                  │  │
│  │                                                        │  │
│  │  1. Strategy Agent  ────►  2. Content Writer         │  │
│  │         │                          │                  │  │
│  │         │                          ▼                  │  │
│  │         │                    3. Creative Agent        │  │
│  │         │                          │                  │  │
│  │         │                          ▼                  │  │
│  │         └────────────►  4. Scheduler Agent           │  │
│  │                                │                      │  │
│  │                                ▼                      │  │
│  │                          5. Posting Agent            │  │
│  │                                │                      │  │
│  │                                ▼                      │  │
│  │                          6. Analytics Agent          │  │
│  │                                │                      │  │
│  │                                ▼                      │  │
│  │                      7. Optimization Agent           │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│   Gemini AI  │ │  Platform   │ │  Database   │
│   (LLM)      │ │  Services   │ │  Layer      │
└──────────────┘ └─────────────┘ └─────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│  Instagram   │ │  Facebook   │ │   Twitter   │
│     API      │ │     API     │ │     API     │
└──────────────┘ └─────────────┘ └─────────────┘
```

## Agent Architecture

### Agent Communication Pattern

```
State Object (Shared Context)
    │
    ├──► Strategy Agent
    │        │
    │        ├──► Goal Analysis
    │        └──► Strategy Output
    │
    ├──► Content Writer Agent
    │        │
    │        ├──► Caption Generation
    │        ├──► Thread Writing
    │        └──► Script Creation
    │
    ├──► Creative Agent
    │        │
    │        ├──► Visual Concepts
    │        └──► Image Prompts
    │
    ├──► Scheduler Agent
    │        │
    │        ├──► Time Optimization
    │        └──► Schedule Generation
    │
    └──► Posting Agent
             │
             ├──► API Integration
             └──► Post Execution
```

## Data Flow

### Campaign Creation Flow

```
1. User submits campaign goal
   ↓
2. Frontend sends POST /api/v1/campaigns
   ↓
3. Campaign Service creates DB record
   ↓
4. Agent Orchestrator executes workflow:
   ├── Strategy Agent: Creates content plan
   ├── Content Writer: Generates captions/scripts
   ├── Creative Agent: Creates visual ideas
   ├── Scheduler: Optimizes posting times
   └── Posting Agent: Prepares posts (pending)
   ↓
5. Posts stored in database (status: pending)
   ↓
6. Celery tasks schedule posts
   ↓
7. Posts published at optimal times
   ↓
8. Analytics Agent tracks performance
   ↓
9. Optimization Agent improves future posts
```

### Posting Flow

```
Scheduled Post Time Arrives
   ↓
Celery Task Triggered
   ↓
Posting Agent Executes
   ├── Validates platform connection
   ├── Formats content for platform
   ├── Calls platform API
   └── Updates post status
   ↓
Post Published
   ↓
Platform Returns Post ID
   ↓
Database Updated
   ↓
Analytics Collection Begins
```

## Database Schema

### PostgreSQL (Relational)

```sql
users
├── id (PK)
├── email
├── username
└── hashed_password

campaigns
├── id (PK)
├── user_id (FK → users)
├── name
├── goal
├── target_platforms (JSON)
├── status
└── strategy_output (JSON)

campaign_posts
├── id (PK)
├── campaign_id (FK → campaigns)
├── platform
├── content_type
├── content (JSON)
├── scheduled_time
├── status
└── platform_post_id

platform_connections
├── id (PK)
├── user_id (FK → users)
├── platform
├── access_token
└── is_active

posts
├── id (PK)
├── user_id (FK → users)
├── platform_connection_id (FK)
├── platform
├── caption
├── hashtags (JSON)
└── metrics (JSON)

analytics
├── id (PK)
├── campaign_id (FK)
├── post_id (FK)
├── metric_type
├── metric_value
└── metric_date
```

## API Design

### RESTful Endpoints

```
Campaigns
├── POST   /api/v1/campaigns           Create campaign
├── GET    /api/v1/campaigns           List campaigns
├── GET    /api/v1/campaigns/{id}      Get campaign
└── POST   /api/v1/campaigns/{id}/execute  Execute posting

Platforms
├── POST   /api/v1/platforms/connect   Connect platform
├── GET    /api/v1/platforms           List connections
└── DELETE /api/v1/platforms/{id}      Disconnect

Analytics
├── GET    /api/v1/analytics/campaign/{id}  Campaign analytics
├── GET    /api/v1/analytics/post/{id}      Post analytics
└── GET    /api/v1/analytics/optimize/{id}  Optimizations

Posts
├── POST   /api/v1/posts               Create/post content
├── GET    /api/v1/posts               List posts
└── GET    /api/v1/posts/{id}          Get post
```

## Security Architecture

### Authentication Flow (Future)

```
User Login
   ↓
JWT Token Generation
   ↓
Token Stored (HttpOnly Cookie)
   ↓
Request with Token
   ↓
Token Validation
   ↓
Authorized Access
```

### API Security

- Environment variables for secrets
- Rate limiting (future)
- CORS configuration
- Input validation (Pydantic)
- SQL injection prevention (SQLAlchemy ORM)

## Scalability Considerations

### Horizontal Scaling

1. **Stateless Backend** - FastAPI instances can scale horizontally
2. **Database Connection Pooling** - SQLAlchemy connection pools
3. **Redis for Caching** - Shared cache across instances
4. **Celery Workers** - Multiple workers for task processing

### Performance Optimization

1. **Async Operations** - FastAPI async endpoints
2. **Background Tasks** - Celery for long-running operations
3. **Database Indexing** - Indexes on frequently queried fields
4. **Response Caching** - Cache analytics results

## Error Handling

### Agent Error Handling

```
Agent Execution
   ↓
Try/Catch Block
   ↓
Error Occurred?
   ├── Yes → Log Error → Return Error State
   └── No  → Continue Workflow
```

### API Error Handling

```
Request Received
   ↓
Validation
   ↓
Valid?
   ├── No  → 422 Validation Error
   └── Yes → Process Request
            ↓
            Error?
            ├── Yes → 500 Server Error + Log
            └── No  → 200 Success Response
```

## Monitoring & Logging

### Logging Strategy

- **Application Logs** - Python logging module
- **API Logs** - FastAPI request logging
- **Agent Logs** - Agent execution logs
- **Error Logs** - Error tracking

### Metrics to Track

- Campaign creation rate
- Post success rate
- API response times
- Agent execution times
- Error rates
- Platform API quotas

## Future Enhancements

1. **Real-time Updates** - WebSocket connections
2. **User Authentication** - JWT-based auth
3. **Image Generation** - DALL-E/Midjourney integration
4. **Video Generation** - Runway/Pika integration
5. **A/B Testing** - Content variant testing
6. **Advanced Analytics** - ML-based insights
7. **Multi-user Support** - Team collaboration

This architecture provides a solid foundation for building and scaling the social media automation system.

