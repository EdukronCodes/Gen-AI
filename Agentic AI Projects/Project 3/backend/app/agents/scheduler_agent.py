"""
Scheduler Agent: Optimizes posting times based on audience and performance data
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent


class SchedulerAgent(BaseAgent):
    """Agent responsible for optimizing posting schedules"""
    
    def __init__(self):
        system_prompt = """You are a Social Media Scheduling Agent.
You analyze audience behavior, engagement patterns, and platform algorithms.
You determine optimal posting times for maximum reach and engagement.
You consider time zones, audience demographics, and historical performance."""
        
        super().__init__(
            name="Scheduler Agent",
            role="Post Timing Optimization",
            system_prompt=system_prompt
        )
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimal posting schedule"""
        platform = context.get("platform", "instagram")
        target_audience = context.get("target_audience", {})
        timezone = context.get("timezone", "UTC")
        duration_days = context.get("duration_days", 7)
        posts_per_day = context.get("posts_per_day", 1)
        historical_data = context.get("historical_data", {})
        
        schedule = await self._generate_schedule(
            platform, target_audience, timezone, duration_days, posts_per_day, historical_data
        )
        
        return {
            "agent": self.name,
            "status": "completed",
            "platform": platform,
            "schedule": schedule
        }
    
    async def _generate_schedule(
        self,
        platform: str,
        target_audience: Dict,
        timezone: str,
        duration_days: int,
        posts_per_day: int,
        historical_data: Dict
    ) -> Dict[str, Any]:
        """Generate optimized posting schedule"""
        
        # Platform-specific best times (default recommendations)
        platform_defaults = {
            "instagram": ["09:00", "13:00", "18:00", "21:00"],
            "facebook": ["08:00", "13:00", "17:00"],
            "twitter": ["08:00", "12:00", "16:00", "20:00"],
            "youtube": ["14:00", "18:00", "22:00"]
        }
        
        prompt = f"""
Create an optimal posting schedule for:

PLATFORM: {platform}
TIMEZONE: {timezone}
DURATION: {duration_days} days
POSTS PER DAY: {posts_per_day}
TARGET AUDIENCE: {target_audience}
HISTORICAL PERFORMANCE: {historical_data}

Consider:
1. Platform-specific peak engagement times
2. Audience timezone and activity patterns
3. Content type (posts perform differently than stories/reels)
4. Day of week patterns (weekday vs weekend)
5. Competitor posting times (avoid overcrowding)

Default platform times: {platform_defaults.get(platform, [])}

Generate a schedule with:
- Specific times for each day
- Day-of-week optimization
- Timezone adjustments
- Reasoning for each time slot

Format as JSON:
{{
    "schedule": [
        {{
            "date": "YYYY-MM-DD",
            "day_of_week": "Monday",
            "times": ["HH:MM", "HH:MM"],
            "reasoning": "why these times"
        }},
        ...
    ],
    "optimal_times_summary": {{
        "weekday": ["HH:MM", "HH:MM"],
        "weekend": ["HH:MM", "HH:MM"]
    }},
    "timezone": "{timezone}",
    "recommendations": ["rec1", "rec2"]
}}
"""
        
        format_description = """Return JSON with schedule array, optimal_times_summary, timezone, and recommendations."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.7)
        
        # Generate actual datetime objects
        if "schedule" in result:
            result["schedule"] = self._convert_to_datetimes(
                result["schedule"], timezone, duration_days
            )
        
        return result
    
    def _convert_to_datetimes(
        self, schedule: List[Dict], timezone: str, duration_days: int
    ) -> List[Dict]:
        """Convert schedule times to datetime objects"""
        # Simplified - in production, use proper timezone handling
        base_date = datetime.now()
        result = []
        
        for day_offset in range(duration_days):
            current_date = base_date + timedelta(days=day_offset)
            day_schedule = schedule[day_offset] if day_offset < len(schedule) else {}
            
            times = day_schedule.get("times", ["12:00"])
            datetime_times = []
            
            for time_str in times:
                try:
                    hour, minute = map(int, time_str.split(":"))
                    dt = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    datetime_times.append(dt.isoformat())
                except:
                    continue
            
            result.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "times": datetime_times,
                **day_schedule
            })
        
        return result
    
    def get_next_best_time(self, platform: str, timezone: str = "UTC") -> datetime:
        """Get the next optimal posting time"""
        # Simplified logic - returns next hour
        now = datetime.now()
        optimal_hours = {
            "instagram": [9, 13, 18, 21],
            "facebook": [8, 13, 17],
            "twitter": [8, 12, 16, 20],
            "youtube": [14, 18, 22]
        }
        
        hours = optimal_hours.get(platform, [12, 18])
        current_hour = now.hour
        
        # Find next optimal hour
        for hour in hours:
            if hour > current_hour:
                return now.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # If past all hours, use first hour tomorrow
        return (now + timedelta(days=1)).replace(hour=hours[0], minute=0, second=0, microsecond=0)

