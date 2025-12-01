"""
Analytics Agent: Tracks and analyzes performance metrics
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent


class AnalyticsAgent(BaseAgent):
    """Agent responsible for tracking and analyzing performance"""
    
    def __init__(self):
        system_prompt = """You are a Social Media Analytics Agent.
You analyze performance metrics, engagement rates, and audience behavior.
You identify trends, patterns, and insights from social media data.
You provide actionable recommendations based on data analysis."""
        
        super().__init__(
            name="Analytics Agent",
            role="Performance Analytics",
            system_prompt=system_prompt
        )
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        platform = context.get("platform", "instagram")
        post_id = context.get("post_id")
        campaign_id = context.get("campaign_id")
        date_range = context.get("date_range", {})
        
        if post_id:
            return await self._analyze_post(post_id, platform)
        elif campaign_id:
            return await self._analyze_campaign(campaign_id, date_range)
        else:
            return await self._get_overview_analytics(platform, date_range)
    
    async def _analyze_post(self, post_id: str, platform: str) -> Dict[str, Any]:
        """Analyze individual post performance"""
        
        prompt = f"""
Analyze the performance of a social media post:

PLATFORM: {platform}
POST ID: {post_id}

Provide analysis including:
1. Engagement rate calculation
2. Reach vs impressions
3. Audience demographics breakdown
4. Peak engagement times
5. Comparison to average performance
6. Key insights
7. Recommendations for similar posts

Format as JSON with engagement_rate, reach, impressions, demographics, insights, and recommendations.
"""
        
        format_description = """Return JSON with engagement_rate, reach, impressions, demographics, peak_times, insights, and recommendations."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.7)
        
        return {
            "agent": self.name,
            "status": "completed",
            "post_id": post_id,
            "platform": platform,
            "analysis": result
        }
    
    async def _analyze_campaign(self, campaign_id: int, date_range: Dict) -> Dict[str, Any]:
        """Analyze campaign performance"""
        
        prompt = f"""
Analyze a social media campaign performance:

CAMPAIGN ID: {campaign_id}
DATE RANGE: {date_range}

Provide comprehensive campaign analysis:
1. Overall engagement metrics
2. Platform performance comparison
3. Content type performance
4. Best performing posts
5. Audience growth
6. Conversion metrics (if applicable)
7. ROI calculation
8. Strategic insights
9. Recommendations for optimization

Format as JSON with overall_metrics, platform_comparison, top_posts, audience_growth, roi, insights, and recommendations.
"""
        
        format_description = """Return JSON with overall_metrics, platform_comparison, top_posts, audience_growth, roi, insights, and recommendations."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.7)
        
        return {
            "agent": self.name,
            "status": "completed",
            "campaign_id": campaign_id,
            "analysis": result
        }
    
    async def _get_overview_analytics(self, platform: str, date_range: Dict) -> Dict[str, Any]:
        """Get overview analytics for platform"""
        
        end_date = date_range.get("end_date", datetime.now())
        start_date = date_range.get("start_date", end_date - timedelta(days=7))
        
        prompt = f"""
Analyze overall social media performance:

PLATFORM: {platform}
START DATE: {start_date}
END DATE: {end_date}

Provide:
1. Total reach and impressions
2. Engagement rate trends
3. Follower growth
4. Best performing content types
5. Optimal posting times analysis
6. Audience insights
7. Platform-specific recommendations

Format as JSON with summary_metrics, trends, top_performers, audience_insights, and recommendations.
"""
        
        format_description = """Return JSON with summary_metrics, trends, top_performers, audience_insights, and recommendations."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.7)
        
        return {
            "agent": self.name,
            "status": "completed",
            "platform": platform,
            "date_range": date_range,
            "analytics": result
        }
    
    def calculate_engagement_rate(self, likes: int, comments: int, shares: int, reach: int) -> float:
        """Calculate engagement rate"""
        if reach == 0:
            return 0.0
        total_engagements = likes + comments + shares
        return (total_engagements / reach) * 100

