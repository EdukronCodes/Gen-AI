"""
Posting Agent: Handles actual posting to social media platforms via APIs
"""
from typing import Dict, Any, Optional
from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.services.platform_services import (
    InstagramService,
    FacebookService,
    TwitterService,
    YouTubeService
)


class PostingAgent(BaseAgent):
    """Agent responsible for posting content to platforms"""
    
    def __init__(self):
        system_prompt = """You are a Social Media Posting Agent.
You handle the technical execution of posting content to platforms.
You manage API calls, handle errors, and ensure successful posting.
You respect rate limits and implement retry logic."""
        
        super().__init__(
            name="Posting Agent",
            role="Content Publishing",
            system_prompt=system_prompt
        )
        
        # Initialize platform services
        self.platform_services = {
            "instagram": InstagramService(),
            "facebook": FacebookService(),
            "twitter": TwitterService(),
            "youtube": YouTubeService()
        }
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Post content to specified platform"""
        platform = context.get("platform", "instagram")
        content = context.get("content", {})
        media_paths = context.get("media_paths", [])
        platform_connection = context.get("platform_connection")
        scheduled_time = context.get("scheduled_time")
        
        if not platform_connection:
            return {
                "agent": self.name,
                "status": "failed",
                "error": "No platform connection provided"
            }
        
        # Check if it's time to post
        if scheduled_time:
            scheduled_dt = datetime.fromisoformat(scheduled_time) if isinstance(scheduled_time, str) else scheduled_time
            now = datetime.now()
            if scheduled_dt > now:
                return {
                    "agent": self.name,
                    "status": "scheduled",
                    "scheduled_time": scheduled_time,
                    "message": f"Post scheduled for {scheduled_time}"
                }
        
        # Get platform service
        service = self.platform_services.get(platform)
        if not service:
            return {
                "agent": self.name,
                "status": "failed",
                "error": f"Unsupported platform: {platform}"
            }
        
        # Post content
        try:
            result = await service.post(
                content=content,
                media_paths=media_paths,
                access_token=platform_connection.get("access_token"),
                **context
            )
            
            return {
                "agent": self.name,
                "status": "completed",
                "platform": platform,
                "platform_post_id": result.get("post_id"),
                "post_url": result.get("post_url"),
                "posted_at": datetime.now().isoformat(),
                "result": result
            }
        except Exception as e:
            return {
                "agent": self.name,
                "status": "failed",
                "platform": platform,
                "error": str(e)
            }
    
    async def schedule_post(
        self,
        platform: str,
        content: Dict,
        scheduled_time: datetime,
        platform_connection: Dict
    ) -> Dict[str, Any]:
        """Schedule a post for later"""
        return {
            "agent": self.name,
            "status": "scheduled",
            "platform": platform,
            "scheduled_time": scheduled_time.isoformat(),
            "message": "Post queued for scheduling"
        }
    
    async def retry_post(
        self,
        platform: str,
        content: Dict,
        platform_connection: Dict,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """Retry posting with exponential backoff"""
        for attempt in range(retry_count):
            try:
                result = await self.execute({
                    "platform": platform,
                    "content": content,
                    "platform_connection": platform_connection
                })
                
                if result["status"] == "completed":
                    return result
                
                # Wait before retry (exponential backoff)
                import asyncio
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                if attempt == retry_count - 1:
                    return {
                        "agent": self.name,
                        "status": "failed",
                        "error": f"Failed after {retry_count} attempts: {str(e)}"
                    }
        
        return {
            "agent": self.name,
            "status": "failed",
            "error": "Max retries exceeded"
        }

