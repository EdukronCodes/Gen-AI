"""
Strategy Agent: Defines goals, content categories, and platform priorities
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent


class StrategyAgent(BaseAgent):
    """Agent responsible for creating content strategy"""
    
    def __init__(self):
        system_prompt = """You are a Strategic Social Media Marketing Agent. 
Your role is to analyze user goals and create comprehensive content strategies.
You decide what to post, when to post, on which platform, and with what approach.
You are analytical, data-driven, and focused on achieving user objectives."""
        
        super().__init__(
            name="Strategy Agent",
            role="Content Strategy Planning",
            system_prompt=system_prompt
        )
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create content strategy based on campaign goals"""
        goal = context.get("goal", "")
        target_platforms = context.get("target_platforms", [])
        duration_days = context.get("duration_days", 7)
        content_themes = context.get("content_themes", [])
        target_audience = context.get("target_audience", {})
        
        prompt = f"""
Create a comprehensive social media content strategy based on the following:

GOAL: {goal}
PLATFORMS: {', '.join(target_platforms)}
DURATION: {duration_days} days
CONTENT THEMES: {', '.join(content_themes) if content_themes else 'Not specified'}
TARGET AUDIENCE: {target_audience}

Provide a detailed strategy including:
1. Content mix per platform (e.g., 2 Instagram posts, 1 reel, 1 story per day)
2. Content categories and themes for each platform
3. Posting frequency per platform
4. Content pillars to focus on
5. Engagement strategy
6. Call-to-action recommendations

Format your response as JSON with this structure:
{{
    "content_plan": {{
        "platform_name": {{
            "post_count": number,
            "content_types": ["type1", "type2"],
            "daily_post_count": number,
            "themes": ["theme1", "theme2"],
            "engagement_tactics": ["tactic1", "tactic2"]
        }}
    }},
    "content_pillars": ["pillar1", "pillar2", "pillar3"],
    "cta_strategy": "description",
    "engagement_strategy": "description",
    "platform_priorities": ["platform1", "platform2"]
}}
"""
        
        format_description = """Return a JSON object with content_plan, content_pillars, cta_strategy, engagement_strategy, and platform_priorities."""
        
        strategy = self.generate_structured(prompt, format_description, temperature=0.8)
        
        return {
            "agent": self.name,
            "status": "completed",
            "output": strategy,
            "recommendations": self._generate_recommendations(strategy)
        }
    
    def _generate_recommendations(self, strategy: Dict[str, Any]) -> list:
        """Generate actionable recommendations from strategy"""
        recommendations = []
        
        if "content_pillars" in strategy:
            recommendations.append(
                f"Focus on {len(strategy['content_pillars'])} core content pillars: "
                f"{', '.join(strategy['content_pillars'][:3])}"
            )
        
        if "platform_priorities" in strategy:
            recommendations.append(
                f"Prioritize platforms in this order: "
                f"{' > '.join(strategy['platform_priorities'])}"
            )
        
        return recommendations

