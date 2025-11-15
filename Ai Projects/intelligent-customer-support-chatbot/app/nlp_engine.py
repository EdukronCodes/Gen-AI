"""
NLP Engine for intent classification, entity extraction, and sentiment analysis
"""
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import re
from config import settings
from app.models import Sentiment

client = OpenAI(api_key=settings.openai_api_key)


class NLPEngine:
    """Natural Language Processing Engine"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intents = self._load_intents()
    
    def _load_intents(self) -> Dict[str, Dict]:
        """Load predefined intents"""
        return {
            "product_inquiry": {
                "description": "Questions about products, features, or specifications",
                "keywords": ["product", "feature", "specification", "what is", "tell me about"],
                "confidence_threshold": 0.7
            },
            "order_status": {
                "description": "Questions about order status, tracking, or delivery",
                "keywords": ["order", "tracking", "delivery", "shipment", "status"],
                "confidence_threshold": 0.7
            },
            "return_refund": {
                "description": "Requests for returns, refunds, or exchanges",
                "keywords": ["return", "refund", "exchange", "cancel", "money back"],
                "confidence_threshold": 0.7
            },
            "technical_support": {
                "description": "Technical issues or troubleshooting requests",
                "keywords": ["problem", "issue", "error", "not working", "broken", "help"],
                "confidence_threshold": 0.7
            },
            "account_management": {
                "description": "Account-related queries",
                "keywords": ["account", "profile", "password", "login", "sign up"],
                "confidence_threshold": 0.7
            },
            "billing_payment": {
                "description": "Billing and payment questions",
                "keywords": ["billing", "payment", "invoice", "charge", "credit card"],
                "confidence_threshold": 0.7
            },
            "general_faq": {
                "description": "General frequently asked questions",
                "keywords": ["how", "what", "when", "where", "why"],
                "confidence_threshold": 0.6
            },
            "complaint": {
                "description": "Customer complaints or dissatisfaction",
                "keywords": ["complaint", "unhappy", "dissatisfied", "poor", "bad"],
                "confidence_threshold": 0.7
            }
        }
    
    def classify_intent(self, message: str) -> Tuple[str, float]:
        """Classify user intent from message"""
        message_lower = message.lower()
        
        # Simple keyword-based classification with scoring
        intent_scores = {}
        for intent_name, intent_data in self.intents.items():
            score = 0
            keyword_matches = sum(1 for keyword in intent_data["keywords"] if keyword in message_lower)
            if keyword_matches > 0:
                score = keyword_matches / len(intent_data["keywords"])
            intent_scores[intent_name] = score
        
        # Use LLM for better classification if available
        if settings.openai_api_key:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an intent classifier. Classify the user message into one of these intents: " + ", ".join(self.intents.keys())},
                        {"role": "user", "content": f"Classify this message: {message}"}
                    ],
                    temperature=0.3,
                    max_tokens=50
                )
                llm_intent = response.choices[0].message.content.strip().lower()
                if llm_intent in intent_scores:
                    intent_scores[llm_intent] += 0.3
            except Exception as e:
                print(f"LLM intent classification failed: {e}")
        
        # Get best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            if confidence >= self.intents[best_intent]["confidence_threshold"]:
                return best_intent, min(confidence, 1.0)
        
        return "general_faq", 0.5
    
    def extract_entities(self, message: str) -> List[Dict[str, str]]:
        """Extract entities from message"""
        entities = []
        
        # Extract order numbers (alphanumeric patterns)
        order_pattern = r'\b[A-Z0-9]{6,}\b'
        order_matches = re.findall(order_pattern, message)
        for match in order_matches:
            entities.append({"type": "order_number", "value": match})
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, message)
        for match in email_matches:
            entities.append({"type": "email", "value": match})
        
        # Extract monetary values
        money_pattern = r'\$[\d,]+\.?\d*'
        money_matches = re.findall(money_pattern, message)
        for match in money_matches:
            entities.append({"type": "amount", "value": match})
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        date_matches = re.findall(date_pattern, message)
        for match in date_matches:
            entities.append({"type": "date", "value": match})
        
        return entities
    
    def analyze_sentiment(self, message: str) -> Tuple[Sentiment, float]:
        """Analyze sentiment of message"""
        positive_words = ["good", "great", "excellent", "amazing", "love", "happy", "satisfied", "thank"]
        negative_words = ["bad", "terrible", "awful", "hate", "angry", "frustrated", "disappointed", "poor"]
        
        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if negative_count > positive_count:
            return Sentiment.NEGATIVE, 0.7
        elif positive_count > negative_count:
            return Sentiment.POSITIVE, 0.7
        else:
            return Sentiment.NEUTRAL, 0.5
    
    def should_escalate(self, message: str, confidence: float, sentiment: Sentiment) -> bool:
        """Determine if conversation should be escalated to human"""
        # Check for explicit escalation keywords
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in settings.auto_escalate_keywords):
            return True
        
        # Check confidence threshold
        if confidence < settings.escalation_confidence_threshold:
            return True
        
        # Check sentiment
        if sentiment == Sentiment.NEGATIVE:
            return True
        
        return False

