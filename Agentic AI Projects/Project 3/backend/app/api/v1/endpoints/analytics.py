"""
Analytics API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from datetime import datetime, timedelta

from app.core.database import get_db
from app.agents.analytics_agent import AnalyticsAgent
from app.agents.optimization_agent import OptimizationAgent

router = APIRouter()
analytics_agent = AnalyticsAgent()
optimization_agent = OptimizationAgent()


@router.get("/campaign/{campaign_id}")
async def get_campaign_analytics(
    campaign_id: int,
    start_date: str = None,
    end_date: str = None,
    db: Session = Depends(get_db)
):
    """Get analytics for a campaign"""
    date_range = {}
    
    if start_date:
        date_range["start_date"] = datetime.fromisoformat(start_date)
    else:
        date_range["start_date"] = datetime.now() - timedelta(days=7)
    
    if end_date:
        date_range["end_date"] = datetime.fromisoformat(end_date)
    else:
        date_range["end_date"] = datetime.now()
    
    context = {
        "campaign_id": campaign_id,
        "date_range": date_range
    }
    
    try:
        result = await analytics_agent.execute(context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/post/{post_id}")
async def get_post_analytics(post_id: str, platform: str = "instagram"):
    """Get analytics for a specific post"""
    context = {
        "post_id": post_id,
        "platform": platform
    }
    
    try:
        result = await analytics_agent.execute(context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/{platform}")
async def get_platform_analytics(
    platform: str,
    start_date: str = None,
    end_date: str = None
):
    """Get analytics for a platform"""
    date_range = {}
    
    if start_date:
        date_range["start_date"] = datetime.fromisoformat(start_date)
    else:
        date_range["start_date"] = datetime.now() - timedelta(days=7)
    
    if end_date:
        date_range["end_date"] = datetime.fromisoformat(end_date)
    else:
        date_range["end_date"] = datetime.now()
    
    context = {
        "platform": platform,
        "date_range": date_range
    }
    
    try:
        result = await analytics_agent.execute(context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/{campaign_id}")
async def get_optimization_recommendations(
    campaign_id: int,
    db: Session = Depends(get_db)
):
    """Get optimization recommendations for a campaign"""
    # Get analytics first
    analytics_context = {
        "campaign_id": campaign_id,
        "date_range": {
            "start_date": datetime.now() - timedelta(days=30),
            "end_date": datetime.now()
        }
    }
    
    analytics_result = await analytics_agent.execute(analytics_context)
    
    # Generate optimizations
    optimization_context = {
        "performance_data": {},
        "analytics_results": analytics_result,
        "campaign_data": {}
    }
    
    try:
        optimization_result = await optimization_agent.execute(optimization_context)
        return {
            "analytics": analytics_result,
            "optimizations": optimization_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

