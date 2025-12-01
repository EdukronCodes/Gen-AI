"""
Content Writer Agent: Writes captions, threads, scripts, and all text content
"""
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent


class ContentWriterAgent(BaseAgent):
    """Agent responsible for generating all text content"""
    
    def __init__(self):
        system_prompt = """You are a Creative Social Media Content Writer Agent.
You write engaging, authentic, and platform-optimized content.
You adapt your tone and style to each platform while maintaining brand voice.
You create captions, threads, scripts, hashtags, and all text content."""
        
        super().__init__(
            name="Content Writer Agent",
            role="Text Content Generation",
            system_prompt=system_prompt
        )
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content based on specifications"""
        platform = context.get("platform", "instagram")
        content_type = context.get("content_type", "post")
        theme = context.get("theme", "")
        topic = context.get("topic", "")
        brand_voice = context.get("brand_voice", "professional yet friendly")
        target_audience = context.get("target_audience", {})
        goal = context.get("goal", "engagement")
        
        content = await self._generate_content(
            platform, content_type, theme, topic, brand_voice, target_audience, goal
        )
        
        return {
            "agent": self.name,
            "status": "completed",
            "platform": platform,
            "content_type": content_type,
            "content": content
        }
    
    async def _generate_content(
        self,
        platform: str,
        content_type: str,
        theme: str,
        topic: str,
        brand_voice: str,
        target_audience: Dict,
        goal: str
    ) -> Dict[str, Any]:
        """Generate content based on platform and type"""
        
        if content_type == "thread" and platform == "twitter":
            return await self._generate_thread(topic, theme, brand_voice, goal)
        elif content_type == "script" and platform == "youtube":
            return await self._generate_video_script(topic, theme, brand_voice, goal)
        elif content_type == "caption":
            return await self._generate_caption(
                platform, topic, theme, brand_voice, target_audience, goal
            )
        else:
            return await self._generate_caption(
                platform, topic, theme, brand_voice, target_audience, goal
            )
    
    async def _generate_caption(
        self,
        platform: str,
        topic: str,
        theme: str,
        brand_voice: str,
        target_audience: Dict,
        goal: str
    ) -> Dict[str, Any]:
        """Generate platform-specific caption"""
        
        platform_guidelines = {
            "instagram": "Use emojis, line breaks, and engaging hooks. Include relevant hashtags.",
            "facebook": "More conversational and longer form. Encourage discussion.",
            "twitter": "Concise, punchy, engaging. Use trending topics when relevant.",
            "youtube": "Descriptive, SEO-optimized. Include keywords and clear descriptions."
        }
        
        prompt = f"""
Write an engaging {platform} caption for the following:

TOPIC: {topic}
THEME: {theme}
BRAND VOICE: {brand_voice}
GOAL: {goal}
PLATFORM GUIDELINES: {platform_guidelines.get(platform, '')}

Generate:
1. A hook (first line that grabs attention)
2. Main caption text
3. Call-to-action (CTA)
4. 10-15 relevant hashtags
5. Suggested emojis (if applicable)

Format as JSON:
{{
    "hook": "attention-grabbing first line",
    "caption": "main caption text with line breaks",
    "cta": "call to action",
    "hashtags": ["hashtag1", "hashtag2"],
    "emoji_suggestions": ["emoji1", "emoji2"],
    "word_count": number,
    "tone": "description of tone used"
}}
"""
        
        format_description = """Return JSON with hook, caption, cta, hashtags, emoji_suggestions, word_count, and tone."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.9)
        
        return result
    
    async def _generate_thread(self, topic: str, theme: str, brand_voice: str, goal: str) -> Dict[str, Any]:
        """Generate Twitter thread"""
        
        prompt = f"""
Write an engaging Twitter thread (6-10 tweets) on the following:

TOPIC: {topic}
THEME: {theme}
BRAND VOICE: {brand_voice}
GOAL: {goal}

Requirements:
- Each tweet should be under 280 characters
- Start with a hook tweet
- Number each tweet (1/N, 2/N, etc.)
- Each tweet should build on the previous
- End with a strong CTA
- Include relevant hashtags in the final tweet

Format as JSON:
{{
    "tweets": [
        {{"number": 1, "text": "tweet content", "character_count": number}},
        ...
    ],
    "thread_summary": "brief summary",
    "hashtags": ["hashtag1", "hashtag2"],
    "total_tweets": number
}}
"""
        
        format_description = """Return JSON with tweets array, thread_summary, hashtags, and total_tweets."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.85)
        
        return result
    
    async def _generate_video_script(
        self, topic: str, theme: str, brand_voice: str, goal: str
    ) -> Dict[str, Any]:
        """Generate YouTube video script"""
        
        prompt = f"""
Write a YouTube video script (for a 3-5 minute video) on:

TOPIC: {topic}
THEME: {theme}
BRAND VOICE: {brand_voice}
GOAL: {goal}

Include:
1. Hook (first 15 seconds)
2. Introduction
3. Main content sections (with timestamps)
4. Conclusion
5. Call-to-action
6. Suggested title (SEO-optimized)
7. Description template
8. Tags/keywords

Format as JSON:
{{
    "title": "video title",
    "hook": "first 15 seconds script",
    "introduction": "introduction script",
    "sections": [
        {{"timestamp": "0:30", "content": "section content", "duration": "30s"}},
        ...
    ],
    "conclusion": "conclusion script",
    "cta": "call to action",
    "description": "full description with timestamps",
    "tags": ["tag1", "tag2"],
    "estimated_duration": "X minutes"
}}
"""
        
        format_description = """Return JSON with title, hook, introduction, sections, conclusion, cta, description, tags, and estimated_duration."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.8)
        
        return result
    
    def generate_hashtags(self, topic: str, platform: str, count: int = 15) -> List[str]:
        """Generate relevant hashtags"""
        
        prompt = f"""
Generate {count} relevant hashtags for {platform} on the topic: {topic}

Include:
- Broad hashtags (high reach)
- Niche hashtags (targeted)
- Trending hashtags (if applicable)
- Brand hashtags

Return as JSON array of hashtag strings (without # symbol).
"""
        
        result = self.generate_structured(
            prompt,
            "Return a JSON array of hashtag strings",
            temperature=0.7
        )
        
        if isinstance(result, dict) and "hashtags" in result:
            return result["hashtags"]
        elif isinstance(result, list):
            return result[:count]
        else:
            return []

