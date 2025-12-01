"""
Agent Orchestrator: Coordinates all agents using LangGraph workflow
"""
from typing import Dict, Any, List, TypedDict
from app.agents.strategy_agent import StrategyAgent
from app.agents.content_writer_agent import ContentWriterAgent
from app.agents.creative_agent import CreativeAgent
from app.agents.scheduler_agent import SchedulerAgent
from app.agents.posting_agent import PostingAgent
from app.agents.analytics_agent import AnalyticsAgent
from app.agents.optimization_agent import OptimizationAgent


class AgentState(TypedDict):
    """State passed between agents"""
    campaign_id: int
    goal: str
    target_platforms: List[str]
    duration_days: int
    content_themes: List[str]
    target_audience: Dict[str, Any]
    strategy_output: Dict[str, Any]
    content_outputs: List[Dict[str, Any]]
    schedule_output: Dict[str, Any]
    posts: List[Dict[str, Any]]
    analytics: Dict[str, Any]
    optimizations: Dict[str, Any]
    errors: List[str]


class AgentOrchestrator:
    """Orchestrates multi-agent workflow"""
    
    def __init__(self):
        self.strategy_agent = StrategyAgent()
        self.content_writer = ContentWriterAgent()
        self.creative_agent = CreativeAgent()
        self.scheduler_agent = SchedulerAgent()
        self.posting_agent = PostingAgent()
        self.analytics_agent = AnalyticsAgent()
        self.optimization_agent = OptimizationAgent()
    
    async def execute_campaign_workflow(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete campaign workflow"""
        
        state: AgentState = {
            "campaign_id": campaign_data.get("campaign_id", 0),
            "goal": campaign_data.get("goal", ""),
            "target_platforms": campaign_data.get("target_platforms", []),
            "duration_days": campaign_data.get("duration_days", 7),
            "content_themes": campaign_data.get("content_themes", []),
            "target_audience": campaign_data.get("target_audience", {}),
            "strategy_output": {},
            "content_outputs": [],
            "schedule_output": {},
            "posts": [],
            "analytics": {},
            "optimizations": {},
            "errors": []
        }
        
        try:
            # Step 1: Strategy Agent
            print("🎯 Running Strategy Agent...")
            strategy_context = {
                "goal": state["goal"],
                "target_platforms": state["target_platforms"],
                "duration_days": state["duration_days"],
                "content_themes": state["content_themes"],
                "target_audience": state["target_audience"]
            }
            strategy_result = await self.strategy_agent.execute(strategy_context)
            state["strategy_output"] = strategy_result.get("output", {})
            
            # Step 2: Generate Content for each platform
            print("✍️ Running Content Writer Agent...")
            content_plan = state["strategy_output"].get("content_plan", {})
            state["content_outputs"] = []
            
            for platform in state["target_platforms"]:
                platform_plan = content_plan.get(platform, {})
                content_types = platform_plan.get("content_types", ["post"])
                themes = platform_plan.get("themes", state["content_themes"])
                
                for content_type in content_types:
                    for theme in themes[:2]:  # Limit to avoid too many generations
                        content_context = {
                            "platform": platform,
                            "content_type": content_type,
                            "theme": theme,
                            "topic": state["goal"],
                            "brand_voice": "professional yet friendly",
                            "target_audience": state["target_audience"],
                            "goal": "engagement"
                        }
                        
                        try:
                            content_result = await self.content_writer.execute(content_context)
                            state["content_outputs"].append(content_result)
                            
                            # Generate visual ideas
                            creative_context = {
                                "platform": platform,
                                "content_type": content_type,
                                "topic": state["goal"],
                                "theme": theme,
                                "brand_style": "modern and clean"
                            }
                            creative_result = await self.creative_agent.execute(creative_context)
                            content_result["visual_content"] = creative_result.get("visual_content", {})
                            
                        except Exception as e:
                            state["errors"].append(f"Content generation error: {str(e)}")
            
            # Step 3: Schedule Agent
            print("⏰ Running Scheduler Agent...")
            schedule_context = {
                "platform": state["target_platforms"][0] if state["target_platforms"] else "instagram",
                "target_audience": state["target_audience"],
                "timezone": "UTC",
                "duration_days": state["duration_days"],
                "posts_per_day": 2,
                "historical_data": {}
            }
            schedule_result = await self.scheduler_agent.execute(schedule_context)
            state["schedule_output"] = schedule_result.get("schedule", {})
            
            # Step 4: Create Posts (without actually posting)
            print("📝 Creating post objects...")
            state["posts"] = []
            schedule = state["schedule_output"].get("schedule", [])
            
            for idx, content_output in enumerate(state["content_outputs"][:10]):  # Limit posts
                if idx < len(schedule):
                    day_schedule = schedule[idx]
                    times = day_schedule.get("times", [])
                    
                    post = {
                        "platform": content_output.get("platform"),
                        "content_type": content_output.get("content_type"),
                        "content": content_output.get("content", {}),
                        "visual_content": content_output.get("visual_content", {}),
                        "scheduled_time": times[0] if times else None,
                        "status": "pending"
                    }
                    state["posts"].append(post)
            
            return {
                "status": "completed",
                "state": state,
                "summary": {
                    "strategy_created": bool(state["strategy_output"]),
                    "content_pieces": len(state["content_outputs"]),
                    "posts_created": len(state["posts"]),
                    "schedule_generated": bool(state["schedule_output"]),
                    "errors": len(state["errors"])
                }
            }
            
        except Exception as e:
            state["errors"].append(f"Workflow error: {str(e)}")
            return {
                "status": "failed",
                "state": state,
                "error": str(e)
            }
    
    async def post_content(self, post_data: Dict[str, Any], platform_connection: Dict[str, Any]) -> Dict[str, Any]:
        """Execute posting workflow"""
        context = {
            "platform": post_data.get("platform"),
            "content": post_data.get("content"),
            "media_paths": post_data.get("media_paths", []),
            "platform_connection": platform_connection,
            "scheduled_time": post_data.get("scheduled_time")
        }
        
        return await self.posting_agent.execute(context)
    
    async def analyze_performance(self, campaign_id: int, date_range: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute analytics workflow"""
        context = {
            "campaign_id": campaign_id,
            "date_range": date_range or {}
        }
        
        analytics_result = await self.analytics_agent.execute(context)
        
        # Run optimization based on analytics
        optimization_context = {
            "performance_data": {},
            "analytics_results": analytics_result,
            "campaign_data": {}
        }
        optimization_result = await self.optimization_agent.execute(optimization_context)
        
        return {
            "analytics": analytics_result,
            "optimizations": optimization_result
        }

