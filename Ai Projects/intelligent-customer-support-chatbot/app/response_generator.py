"""
Response generation using LLM and knowledge base
"""
from typing import List, Dict, Optional
from openai import OpenAI
from config import settings
from app.knowledge_base import KnowledgeBase
from app.nlp_engine import NLPEngine


class ResponseGenerator:
    """Generate intelligent responses using LLM and knowledge base"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.knowledge_base = KnowledgeBase()
        self.nlp_engine = NLPEngine()
    
    def generate_response(
        self,
        message: str,
        intent: str,
        conversation_history: Optional[List[Dict]] = None,
        context: Optional[Dict] = None
    ) -> str:
        """Generate response based on user message and intent"""
        
        # Search knowledge base for relevant information
        kb_results = self.knowledge_base.search(message, limit=3)
        kb_context = ""
        if kb_results:
            kb_context = "\n\nRelevant Information:\n"
            for result in kb_results:
                kb_context += f"- {result['title']}: {result['content']}\n"
        
        # Build conversation history context
        history_context = ""
        if conversation_history:
            recent_messages = conversation_history[-5:]  # Last 5 messages
            history_context = "\n\nConversation History:\n"
            for msg in recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_context += f"{role.capitalize()}: {content}\n"
        
        # Build system prompt
        system_prompt = f"""You are a helpful and empathetic customer support chatbot. 
Your goal is to assist customers with their inquiries in a friendly, professional manner.

Intent: {intent}
{history_context}

Guidelines:
- Be concise but thorough
- Use the provided knowledge base information when relevant
- If you don't know something, admit it and offer to connect them with a human agent
- Maintain a friendly and professional tone
- If the customer seems frustrated, show empathy
"""
        
        # Build user prompt
        user_prompt = f"{message}\n\n{kb_context}"
        
        try:
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            if kb_results:
                return f"Based on our knowledge base: {kb_results[0]['content']}"
            else:
                return "I apologize, but I'm having trouble processing your request. Would you like me to connect you with a human agent?"
    
    def generate_escalation_message(self) -> str:
        """Generate message for human handoff"""
        return """I understand you'd like to speak with a human agent. I'm connecting you with one of our support specialists who will be able to assist you further. 
        
In the meantime, could you provide a brief description of your issue so I can prepare our agent with the necessary context?"""

