"""
Google Gemini API client for all LLM tasks
"""
import google.generativeai as genai
from typing import List, Dict, Optional
from app.core.config import settings


class GeminiClient:
    """Client for interacting with Google Gemini API"""
    
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
    
    def generate_text(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate text using Gemini"""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            if system_instruction:
                model = genai.GenerativeModel(
                    'gemini-pro',
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
            raise
    
    def generate_structured_output(
        self,
        prompt: str,
        format_description: str,
        temperature: float = 0.7
    ) -> Dict:
        """Generate structured JSON output"""
        structured_prompt = f"""
{format_description}

{prompt}

Please provide your response in the exact format specified above.
"""
        response_text = self.generate_text(structured_prompt, temperature)
        # Parse JSON from response
        import json
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"raw_response": response_text}
    
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """Analyze image using Gemini Vision"""
        try:
            import PIL.Image
            img = PIL.Image.open(image_path)
            response = self.vision_model.generate_content([prompt, img])
            return response.text
        except Exception as e:
            print(f"Gemini Vision Error: {str(e)}")
            raise


# Global instance
gemini_client = GeminiClient()

