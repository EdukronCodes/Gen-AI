"""
Optimization Agent: Improves future posts based on performance data
"""
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent


class OptimizationAgent(BaseAgent):
    """Agent responsible for optimizing future content based on analytics"""
    
    def __init__(self):
        system_prompt = """You are a Social Media Optimization Agent.
You analyze performance data and identify what works best.
You provide specific recommendations to improve future content.
You learn from successful and unsuccessful posts to optimize strategy."""
        
        super().__init__(
            name="Optimization Agent",
            role="Performance Optimization",
            system_prompt=system_prompt
        )
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        performance_data = context.get("performance_data", {})
        analytics_results = context.get("analytics_results", {})
        campaign_data = context.get("campaign_data", {})
        
        recommendations = await self._generate_optimizations(
            performance_data, analytics_results, campaign_data
        )
        
        return {
            "agent": self.name,
            "status": "completed",
            "recommendations": recommendations,
            "optimized_strategy": await self._create_optimized_strategy(recommendations)
        }
    
    async def _generate_optimizations(
        self,
        performance_data: Dict,
        analytics_results: Dict,
        campaign_data: Dict
    ) -> Dict[str, Any]:
        """Generate optimization recommendations"""
        
        prompt = f"""
Analyze performance data and generate optimization recommendations:

PERFORMANCE DATA: {performance_data}
ANALYTICS RESULTS: {analytics_results}
CAMPAIGN DATA: {campaign_data}

Identify:
1. What's working (high performing elements)
2. What's not working (low performing elements)
3. Content optimization opportunities
4. Timing optimizations
5. Hashtag optimizations
6. Visual/content style improvements
7. Audience targeting refinements
8. Platform-specific optimizations

Provide specific, actionable recommendations.

Format as JSON:
{{
    "whats_working": [
        {{"element": "description", "impact": "high/medium/low", "recommendation": "keep doing this"}},
        ...
    ],
    "whats_not_working": [
        {{"element": "description", "issue": "problem", "fix": "recommendation"}},
        ...
    ],
    "content_optimizations": [
        {{"area": "hooks", "current": "state", "optimized": "recommendation"}},
        ...
    ],
    "timing_optimizations": {{
        "current": "current pattern",
        "optimized": "better times",
        "reasoning": "why"
    }},
    "hashtag_optimizations": {{
        "current_performance": "analysis",
        "recommended_hashtags": ["tag1", "tag2"],
        "strategy": "approach"
    }},
    "visual_optimizations": [
        {{"element": "description", "optimization": "recommendation"}},
        ...
    ],
    "audience_optimizations": [
        {{"aspect": "description", "optimization": "recommendation"}},
        ...
    ],
    "priority_actions": [
        "action1",
        "action2",
        "action3"
    ]
}}
"""
        
        format_description = """Return JSON with whats_working, whats_not_working, content_optimizations, timing_optimizations, hashtag_optimizations, visual_optimizations, audience_optimizations, and priority_actions."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.8)
        
        return result
    
    async def _create_optimized_strategy(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized strategy based on recommendations"""
        
        prompt = f"""
Based on these optimization recommendations:

{recommendations}

Create an updated, optimized content strategy that incorporates these improvements.

Include:
1. Updated content guidelines
2. Optimized posting schedule
3. Improved hashtag strategy
4. Enhanced visual guidelines
5. Refined audience targeting
6. Platform-specific optimizations

Format as JSON with guidelines, schedule, hashtags, visuals, targeting, and platform_strategies.
"""
        
        format_description = """Return JSON with guidelines, schedule, hashtags, visuals, targeting, and platform_strategies."""
        
        return self.generate_structured(prompt, format_description, temperature=0.75)

