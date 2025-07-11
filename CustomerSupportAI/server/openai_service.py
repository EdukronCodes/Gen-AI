import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

logger = logging.getLogger(__name__)

@dataclass
class ChatContext:
    userName: str
    lastOrder: Optional[str] = None
    location: Optional[str] = None
    preferredCategory: Optional[str] = None
    chatHistory: Optional[List[Dict[str, Any]]] = None

@dataclass
class ChatResponse:
    message: str
    intent: str
    confidence: float
    suggestedActions: Optional[List[str]] = None

class OpenAIService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        self.available = False
        
        if openai_available and api_key and api_key != "your-api-key-here":
            try:
                self.client = OpenAI(api_key=api_key)
                self.available = True
                logger.info("OpenAI service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.available = False
        else:
            logger.warning("OpenAI service not available - using fallback mode")
    
    def is_available(self) -> bool:
        return self.available
    
    def generate_chat_response(self, user_message: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered chat response with context awareness"""
        
        if not self.available:
            return self._generate_fallback_response(user_message, user_context)
        
        try:
            # Build context-aware prompt
            context_prompt = self._build_context_prompt(user_context)
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an AI customer support assistant for an online retail store. 
                        You are helpful, friendly, and knowledgeable about products, orders, and policies.
                        
                        Current customer context:
                        {context_prompt}
                        
                        Respond naturally and helpfully. Classify the intent of the user's message and provide appropriate suggestions.
                        
                        Return your response in JSON format with these fields:
                        - message: Your helpful response to the customer
                        - intent: The main intent (product_info, order_tracking, returns, recommendations, general)
                        - confidence: Confidence score between 0 and 1
                        - suggestedActions: Array of suggested next steps (optional)
                        
                        Keep responses concise but helpful. Use the customer's name when appropriate."""
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                "message": result.get("message", "I'm here to help! How can I assist you today?"),
                "intent": result.get("intent", "general"),
                "confidence": min(max(result.get("confidence", 0.8), 0.0), 1.0),
                "suggestedActions": result.get("suggestedActions", [])
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._generate_fallback_response(user_message, user_context)
    
    def generate_product_recommendations(self, user_context: Dict[str, Any], available_products: List[Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered product recommendations"""
        
        if not self.available:
            return self._generate_fallback_recommendations(user_context, available_products)
        
        try:
            # Build product context
            products_info = []
            for product in available_products[:10]:  # Limit to avoid token limits
                products_info.append({
                    "id": product.id,
                    "name": product.name,
                    "description": product.description,
                    "price": product.price,
                    "category": product.category
                })
            
            context_prompt = self._build_context_prompt(user_context)
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an AI product recommendation engine for an online retail store.
                        
                        Customer context:
                        {context_prompt}
                        
                        Available products:
                        {json.dumps(products_info, indent=2)}
                        
                        Based on the customer's preferences and history, recommend 3-5 products that would be most relevant.
                        
                        Return recommendations in JSON format as an array of objects with these fields:
                        - productId: The product ID
                        - name: Product name
                        - reason: Brief reason why this product is recommended
                        - priority: Number from 1-5 (5 being highest priority)
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Please recommend products for {user_context.get('userName', 'this customer')}"
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=800,
                temperature=0.6
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("recommendations", [])
            
        except Exception as e:
            logger.error(f"OpenAI recommendations error: {e}")
            return self._generate_fallback_recommendations(user_context, available_products)
    
    def _build_context_prompt(self, user_context: Dict[str, Any]) -> str:
        """Build context prompt from user data"""
        context_parts = []
        
        if user_context.get('userName'):
            context_parts.append(f"Customer name: {user_context['userName']}")
        
        if user_context.get('location'):
            context_parts.append(f"Location: {user_context['location']}")
        
        if user_context.get('preferredCategory'):
            context_parts.append(f"Preferred category: {user_context['preferredCategory']}")
        
        if user_context.get('chatHistory'):
            context_parts.append("Recent chat history:")
            for chat in user_context['chatHistory'][-3:]:  # Last 3 exchanges
                context_parts.append(f"- Customer: {chat['message']}")
                context_parts.append(f"- Assistant: {chat['response']}")
        
        return "\n".join(context_parts) if context_parts else "No additional context available"
    
    def _generate_fallback_response(self, user_message: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rule-based fallback response"""
        message_lower = user_message.lower()
        user_name = user_context.get('userName', 'there')
        
        # Intent detection based on keywords
        if any(keyword in message_lower for keyword in ['order', 'track', 'shipping', 'delivery']):
            return {
                "message": f"Hi {user_name}! I'd be happy to help you with your order. Could you please provide your order ID or confirm your email address to proceed?",
                "intent": "order_tracking",
                "confidence": 0.95,
                "suggestedActions": ["Provide Order ID", "Confirm Email Address"]
            }
        
        elif any(keyword in message_lower for keyword in ['return', 'refund', 'exchange']):
            return {
                "message": f"Hi {user_name}! I can help you with returns and refunds. Our return policy allows returns within 30 days of purchase. What specific item would you like to return?",
                "intent": "returns",
                "confidence": 0.9,
                "suggestedActions": ["Check Return Policy", "Start Return Process"]
            }
        
        elif any(keyword in message_lower for keyword in ['recommend', 'suggest', 'best', 'popular']):
            preferred_category = user_context.get('preferredCategory', 'electronics')
            return {
                "message": f"Sure, {user_name}! Since you prefer {preferred_category}, here are a few recommendations based on your interests: 1. Noise Cancelling Headphones - $199, 2. Smart Home Assistant - $129, 3. Wireless Earbuds - $99. Let me know if you need more details on any of these!",
                "intent": "recommendations",
                "confidence": 0.95,
                "suggestedActions": ["viewProductDetails", "addToCart"]
            }
        
        elif any(keyword in message_lower for keyword in ['product', 'item', 'buy', 'purchase', 'price']):
            return {
                "message": f"Hi {user_name}! I'd be happy to help you find the perfect product. What specific type of item are you looking for? I can help you with product details, pricing, and availability.",
                "intent": "product_info",
                "confidence": 0.85,
                "suggestedActions": ["Browse Products", "Search Specific Item"]
            }
        
        else:
            return {
                "message": f"Hello {user_name}! I'm here to help with any questions about our products, orders, returns, or recommendations. How can I assist you today?",
                "intent": "general",
                "confidence": 0.8,
                "suggestedActions": ["Browse Products", "Track Order", "Contact Support"]
            }
    
    def _generate_fallback_recommendations(self, user_context: Dict[str, Any], available_products: List[Any]) -> List[Dict[str, Any]]:
        """Generate rule-based product recommendations"""
        preferred_category = user_context.get('preferredCategory', '').lower()
        
        # Filter products by preferred category if available
        if preferred_category:
            matching_products = [p for p in available_products if p.category.lower() == preferred_category]
        else:
            matching_products = available_products
        
        # Take first 3 products as recommendations
        recommendations = []
        for i, product in enumerate(matching_products[:3]):
            recommendations.append({
                "productId": product.id,
                "name": product.name,
                "reason": f"Based on your interest in {product.category}",
                "priority": 5 - i  # Higher priority for first items
            })
        
        return recommendations