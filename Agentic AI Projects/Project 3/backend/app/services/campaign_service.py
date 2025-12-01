"""
Campaign service for managing campaigns
"""
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from app.models.campaign import Campaign, CampaignPost, CampaignStatus
from app.agents.agent_orchestrator import AgentOrchestrator
from datetime import datetime, timedelta


class CampaignService:
    """Service for campaign management"""
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
    
    async def create_campaign(self, db: Session, campaign_data: Dict[str, Any]) -> Campaign:
        """Create and execute campaign workflow"""
        
        # Create campaign record
        campaign = Campaign(
            user_id=campaign_data["user_id"],
            name=campaign_data.get("name", "New Campaign"),
            description=campaign_data.get("description", ""),
            goal=campaign_data["goal"],
            target_platforms=campaign_data.get("target_platforms", []),
            duration_days=campaign_data.get("duration_days", 7),
            content_themes=campaign_data.get("content_themes", []),
            target_audience=campaign_data.get("target_audience", {}),
            status=CampaignStatus.DRAFT
        )
        
        db.add(campaign)
        db.commit()
        db.refresh(campaign)
        
        # Execute agent workflow
        workflow_data = {
            "campaign_id": campaign.id,
            **campaign_data
        }
        
        workflow_result = await self.orchestrator.execute_campaign_workflow(workflow_data)
        
        if workflow_result["status"] == "completed":
            # Store strategy output
            campaign.strategy_output = workflow_result["state"].get("strategy_output", {})
            campaign.status = CampaignStatus.ACTIVE
            
            # Create campaign posts
            posts = workflow_result["state"].get("posts", [])
            for post_data in posts:
                campaign_post = CampaignPost(
                    campaign_id=campaign.id,
                    platform=post_data.get("platform"),
                    content_type=post_data.get("content_type"),
                    content=post_data.get("content", {}),
                    scheduled_time=datetime.fromisoformat(post_data["scheduled_time"]) if post_data.get("scheduled_time") else None,
                    status="pending"
                )
                db.add(campaign_post)
            
            # Set dates
            campaign.start_date = datetime.now()
            campaign.end_date = datetime.now() + timedelta(days=campaign.duration_days)
        
        db.commit()
        db.refresh(campaign)
        
        return campaign
    
    async def execute_campaign_posting(
        self,
        db: Session,
        campaign_id: int,
        platform_connections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute posting for a campaign"""
        
        campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
        if not campaign:
            raise ValueError("Campaign not found")
        
        posts = db.query(CampaignPost).filter(
            CampaignPost.campaign_id == campaign_id,
            CampaignPost.status == "pending"
        ).all()
        
        results = []
        
        for post in posts:
            platform = post.platform
            connection = platform_connections.get(platform)
            
            if not connection:
                continue
            
            post_data = {
                "platform": platform,
                "content": post.content,
                "scheduled_time": post.scheduled_time.isoformat() if post.scheduled_time else None
            }
            
            result = await self.orchestrator.post_content(post_data, connection)
            
            if result["status"] == "completed":
                post.status = "posted"
                post.platform_post_id = result.get("platform_post_id")
                post.posted_at = datetime.fromisoformat(result["posted_at"])
            
            results.append(result)
        
        db.commit()
        
        return {
            "campaign_id": campaign_id,
            "results": results,
            "successful": sum(1 for r in results if r["status"] == "completed")
        }
    
    def get_campaign(self, db: Session, campaign_id: int) -> Optional[Campaign]:
        """Get campaign by ID"""
        return db.query(Campaign).filter(Campaign.id == campaign_id).first()
    
    def get_campaign_posts(
        self,
        db: Session,
        campaign_id: int,
        status: Optional[str] = None
    ) -> List[CampaignPost]:
        """Get campaign posts"""
        query = db.query(CampaignPost).filter(CampaignPost.campaign_id == campaign_id)
        
        if status:
            query = query.filter(CampaignPost.status == status)
        
        return query.all()

