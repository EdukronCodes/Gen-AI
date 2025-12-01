"""
Celery tasks for scheduled posting
"""
from celery_app.celery import celery_app
from datetime import datetime
from app.services.campaign_service import CampaignService
from app.core.database import SessionLocal
from app.models.campaign import CampaignPost, Campaign
from app.agents.posting_agent import PostingAgent
from app.models.platform import PlatformConnection
import asyncio


@celery_app.task
def schedule_post_task(post_id: int, platform_connection_id: int):
    """Task to post content at scheduled time"""
    db = SessionLocal()
    posting_agent = PostingAgent()
    
    try:
        # Get post
        post = db.query(CampaignPost).filter(CampaignPost.id == post_id).first()
        if not post or post.status != "pending":
            return {"status": "skipped", "reason": "Post not found or already processed"}
        
        # Get platform connection
        connection = db.query(PlatformConnection).filter(
            PlatformConnection.id == platform_connection_id
        ).first()
        
        if not connection:
            return {"status": "failed", "reason": "Platform connection not found"}
        
        # Post content
        context = {
            "platform": post.platform,
            "content": post.content,
            "media_paths": post.media_paths or [],
            "platform_connection": {
                "access_token": connection.access_token,
                "id": connection.id,
                "user_id": connection.user_id
            }
        }
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(posting_agent.execute(context))
        loop.close()
        
        if result["status"] == "completed":
            post.status = "posted"
            post.platform_post_id = result.get("platform_post_id")
            post.posted_at = datetime.now()
            db.commit()
        
        return result
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()


@celery_app.task
def process_scheduled_posts():
    """Process all posts scheduled for current time"""
    db = SessionLocal()
    
    try:
        now = datetime.now()
        
        # Find posts scheduled for now (within 5 minute window)
        from datetime import timedelta
        time_window_start = now - timedelta(minutes=5)
        time_window_end = now + timedelta(minutes=5)
        
        posts = db.query(CampaignPost).filter(
            CampaignPost.status == "pending",
            CampaignPost.scheduled_time >= time_window_start,
            CampaignPost.scheduled_time <= time_window_end
        ).all()
        
        results = []
        
        for post in posts:
            # Get platform connection
            campaign = db.query(Campaign).filter(Campaign.id == post.campaign_id).first()
            if not campaign:
                continue
            
            connection = db.query(PlatformConnection).filter(
                PlatformConnection.user_id == campaign.user_id,
                PlatformConnection.platform == post.platform,
                PlatformConnection.is_active == True
            ).first()
            
            if connection:
                # Schedule post task
                schedule_post_task.delay(post.id, connection.id)
                results.append({"post_id": post.id, "status": "scheduled"})
        
        return {"processed": len(results), "results": results}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


@celery_app.task
def update_analytics_task(campaign_id: int):
    """Task to update analytics for a campaign"""
    db = SessionLocal()
    
    try:
        # Get campaign posts
        posts = db.query(CampaignPost).filter(
            CampaignPost.campaign_id == campaign_id,
            CampaignPost.status == "posted"
        ).all()
        
        # Update analytics (placeholder - implement actual analytics fetching)
        return {
            "campaign_id": campaign_id,
            "posts_analyzed": len(posts),
            "status": "completed"
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        db.close()


# Periodic task schedule
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "process-scheduled-posts": {
        "task": "celery_app.tasks.process_scheduled_posts",
        "schedule": crontab(minute="*/5"),  # Every 5 minutes
    },
    "update-analytics": {
        "task": "celery_app.tasks.update_analytics_task",
        "schedule": crontab(hour="*/6"),  # Every 6 hours
    },
}

