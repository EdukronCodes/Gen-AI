"""
Health Check Endpoints
"""
from fastapi import APIRouter
from app.core.database import get_db, get_redis, get_mongo_db

router = APIRouter()


@router.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Agentic AI IT Help Desk"
    }


@router.get("/detailed")
async def detailed_health():
    """Detailed health check with dependencies"""
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "redis": "unknown",
        "mongodb": "unknown"
    }
    
    # Check PostgreSQL
    try:
        db = next(get_db())
        db.execute("SELECT 1")
        health_status["database"] = "connected"
    except:
        health_status["database"] = "disconnected"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        redis = get_redis()
        redis.ping()
        health_status["redis"] = "connected"
    except:
        health_status["redis"] = "disconnected"
        health_status["status"] = "degraded"
    
    # Check MongoDB
    try:
        mongo = get_mongo_db()
        mongo.client.admin.command('ping')
        health_status["mongodb"] = "connected"
    except:
        health_status["mongodb"] = "disconnected"
        health_status["status"] = "degraded"
    
    return health_status


