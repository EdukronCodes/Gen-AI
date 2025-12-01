"""
Campaign API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel

from app.core.database import get_db
from app.services.campaign_service import CampaignService
from app.models.campaign import Campaign, CampaignPost

router = APIRouter()
campaign_service = CampaignService()


class CampaignCreate(BaseModel):
    user_id: int
    name: str
    description: str = ""
    goal: str
    target_platforms: List[str]
    duration_days: int = 7
    content_themes: List[str] = []
    target_audience: Dict[str, Any] = {}


class CampaignResponse(BaseModel):
    id: int
    name: str
    description: str
    goal: str
    status: str
    target_platforms: List[str]
    duration_days: int
    content_themes: List[str]
    strategy_output: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True


@router.post("/", response_model=CampaignResponse)
async def create_campaign(
    campaign: CampaignCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new campaign and execute agent workflow"""
    try:
        campaign_data = campaign.dict()
        db_campaign = await campaign_service.create_campaign(db, campaign_data)
        return db_campaign
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[CampaignResponse])
def get_campaigns(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all campaigns"""
    campaigns = db.query(Campaign).offset(skip).limit(limit).all()
    return campaigns


@router.get("/{campaign_id}", response_model=CampaignResponse)
def get_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Get campaign by ID"""
    campaign = campaign_service.get_campaign(db, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign


@router.get("/{campaign_id}/posts")
def get_campaign_posts(
    campaign_id: int,
    status: str = None,
    db: Session = Depends(get_db)
):
    """Get posts for a campaign"""
    posts = campaign_service.get_campaign_posts(db, campaign_id, status)
    return posts


@router.post("/{campaign_id}/execute")
async def execute_campaign(
    campaign_id: int,
    platform_connections: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Execute campaign posting"""
    try:
        result = await campaign_service.execute_campaign_posting(
            db, campaign_id, platform_connections
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

