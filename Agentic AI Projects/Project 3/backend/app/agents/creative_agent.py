"""
Creative Agent: Generates images, thumbnails, and visual content descriptions
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent


class CreativeAgent(BaseAgent):
    """Agent responsible for generating visual content ideas and descriptions"""
    
    def __init__(self):
        system_prompt = """You are a Creative Visual Content Agent.
You generate detailed visual content ideas, image descriptions, and design concepts.
You create prompts for image generation and describe visual elements for social media posts.
You understand platform-specific visual requirements and trends."""
        
        super().__init__(
            name="Creative Agent",
            role="Visual Content Generation",
            system_prompt=system_prompt
        )
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visual content ideas and descriptions"""
        platform = context.get("platform", "instagram")
        content_type = context.get("content_type", "post")
        topic = context.get("topic", "")
        theme = context.get("theme", "")
        brand_style = context.get("brand_style", "modern and clean")
        
        visual_content = await self._generate_visual_ideas(
            platform, content_type, topic, theme, brand_style
        )
        
        return {
            "agent": self.name,
            "status": "completed",
            "platform": platform,
            "content_type": content_type,
            "visual_content": visual_content
        }
    
    async def _generate_visual_ideas(
        self,
        platform: str,
        content_type: str,
        topic: str,
        theme: str,
        brand_style: str
    ) -> Dict[str, Any]:
        """Generate visual content ideas"""
        
        platform_specs = {
            "instagram": {
                "post": "1080x1080px square",
                "reel": "1080x1920px vertical",
                "story": "1080x1920px vertical"
            },
            "facebook": {
                "post": "1200x630px landscape",
                "cover": "1200x630px"
            },
            "twitter": {
                "post": "1200x675px landscape or 1200x1200px square"
            },
            "youtube": {
                "thumbnail": "1280x720px",
                "channel_art": "2560x1440px"
            }
        }
        
        specs = platform_specs.get(platform, {}).get(content_type, "standard")
        
        prompt = f"""
Create visual content ideas for:

PLATFORM: {platform}
CONTENT TYPE: {content_type}
TOPIC: {topic}
THEME: {theme}
BRAND STYLE: {brand_style}
DIMENSIONS: {specs}

Generate:
1. Visual concept description
2. Color palette recommendations
3. Image generation prompt (detailed)
4. Text overlay suggestions
5. Composition guidelines
6. Style references

Format as JSON:
{{
    "visual_concept": "detailed description",
    "color_palette": ["#hex1", "#hex2", "#hex3"],
    "image_prompt": "detailed prompt for AI image generation",
    "text_overlay": {{
        "headline": "text suggestion",
        "subtext": "text suggestion",
        "font_style": "suggested font"
    }},
    "composition": "composition guidelines",
    "style_references": ["style1", "style2"],
    "dimensions": "{specs}"
}}
"""
        
        format_description = """Return JSON with visual_concept, color_palette, image_prompt, text_overlay, composition, style_references, and dimensions."""
        
        result = self.generate_structured(prompt, format_description, temperature=0.9)
        
        return result
    
    def generate_thumbnail_idea(self, video_title: str, topic: str) -> Dict[str, Any]:
        """Generate YouTube thumbnail idea"""
        
        prompt = f"""
Create a YouTube thumbnail design for:

TITLE: {video_title}
TOPIC: {topic}

Generate:
1. Thumbnail concept (high contrast, attention-grabbing)
2. Color scheme
3. Text overlay (title text)
4. Visual elements
5. Face/character positioning (if applicable)
6. Detailed image generation prompt

Format as JSON with concept, colors, text_overlay, visual_elements, positioning, and image_prompt.
"""
        
        format_description = """Return JSON with concept, colors, text_overlay, visual_elements, positioning, and image_prompt."""
        
        return self.generate_structured(prompt, format_description, temperature=0.85)

