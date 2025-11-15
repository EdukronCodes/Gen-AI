"""
Main FastAPI application for the Intelligent Customer Support Chatbot
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import redis

from config import settings
from app.database import get_db, init_db, get_redis
from app.models import (
    ConversationRequest,
    ConversationResponse,
    Conversation,
    Message,
    SearchRequest,
    SearchResponse,
    KnowledgeArticle
)
from app.nlp_engine import NLPEngine
from app.response_generator import ResponseGenerator
from app.knowledge_base import KnowledgeBase
from app.conversation_manager import ConversationManager

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Intelligent Customer Support Chatbot API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
nlp_engine = NLPEngine()
response_generator = ResponseGenerator()
knowledge_base = KnowledgeBase()
conversation_manager = ConversationManager()


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intelligent Customer Support Chatbot API",
        "version": settings.app_version,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chatbot-api"}


@app.post("/api/v1/conversations", response_model=Conversation)
async def create_conversation(
    user_id: Optional[str] = None,
    channel: str = "web",
    db: Session = Depends(get_db)
):
    """Create a new conversation"""
    conversation = conversation_manager.create_conversation(
        db=db,
        user_id=user_id,
        channel=channel
    )
    return conversation


@app.post("/api/v1/conversations/{conversation_id}/messages", response_model=ConversationResponse)
async def send_message(
    conversation_id: str,
    request: ConversationRequest,
    db: Session = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis)
):
    """Send a message and get response"""
    try:
        # Get or create conversation
        conversation = conversation_manager.get_conversation(db, conversation_id)
        if not conversation:
            conversation = conversation_manager.create_conversation(
                db=db,
                conversation_id=conversation_id,
                user_id=request.user_id
            )
        
        # Check message length
        if len(request.message) > settings.max_message_length:
            raise HTTPException(
                status_code=400,
                detail=f"Message exceeds maximum length of {settings.max_message_length} characters"
            )
        
        # Analyze message
        intent, confidence = nlp_engine.classify_intent(request.message)
        entities = nlp_engine.extract_entities(request.message)
        sentiment, sentiment_score = nlp_engine.analyze_sentiment(request.message)
        
        # Get conversation history
        history = conversation_manager.get_conversation_history(db, conversation_id)
        
        # Generate response
        response_text = response_generator.generate_response(
            message=request.message,
            intent=intent,
            conversation_history=history
        )
        
        # Check if escalation is needed
        should_escalate = nlp_engine.should_escalate(
            request.message,
            confidence,
            sentiment
        )
        
        if should_escalate:
            response_text = response_generator.generate_escalation_message()
            conversation_manager.update_conversation_status(db, conversation_id, "escalated")
        
        # Save messages
        conversation_manager.save_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=request.message,
            metadata={
                "intent": intent,
                "entities": entities,
                "sentiment": sentiment.value,
                "confidence": confidence
            }
        )
        
        conversation_manager.save_message(
            db=db,
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            metadata={
                "intent": intent,
                "confidence": confidence
            }
        )
        
        # Update conversation
        conversation_manager.update_conversation_timestamp(db, conversation_id)
        
        return ConversationResponse(
            response=response_text,
            conversation_id=conversation_id,
            confidence=confidence,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            should_escalate=should_escalate,
            metadata={
                "sentiment_score": sentiment_score
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.get("/api/v1/conversations/{conversation_id}", response_model=List[Message])
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get conversation history"""
    messages = conversation_manager.get_conversation_history(db, conversation_id)
    return messages


@app.delete("/api/v1/conversations/{conversation_id}")
async def end_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """End a conversation"""
    success = conversation_manager.update_conversation_status(db, conversation_id, "ended")
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation ended", "conversation_id": conversation_id}


@app.post("/api/v1/knowledge/search", response_model=SearchResponse)
async def search_knowledge(request: SearchRequest):
    """Search knowledge base"""
    results = knowledge_base.search(
        query=request.query,
        category=request.category,
        limit=request.limit
    )
    
    confidence = results[0]["similarity"] if results else 0.0
    
    return SearchResponse(
        results=results,
        confidence=confidence,
        query=request.query
    )


@app.post("/api/v1/knowledge/articles", response_model=KnowledgeArticle)
async def add_knowledge_article(
    title: str,
    content: str,
    category: str,
    tags: Optional[List[str]] = None
):
    """Add article to knowledge base"""
    article_id = knowledge_base.add_article(
        title=title,
        content=content,
        category=category,
        tags=tags or []
    )
    
    return KnowledgeArticle(
        article_id=article_id,
        title=title,
        content=content,
        category=category,
        tags=tags or []
    )


@app.get("/api/v1/knowledge/articles/{article_id}", response_model=KnowledgeArticle)
async def get_knowledge_article(article_id: str):
    """Get article from knowledge base"""
    article = knowledge_base.get_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return KnowledgeArticle(**article)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)

