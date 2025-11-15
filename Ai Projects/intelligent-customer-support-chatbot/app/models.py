"""
Data models for the chatbot application
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    ENDED = "ended"
    ESCALATED = "escalated"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    channel: str = "web"
    status: ConversationStatus = ConversationStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ConversationRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    user_id: Optional[str] = None


class ConversationResponse(BaseModel):
    response: str
    conversation_id: str
    confidence: float
    intent: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    sentiment: Optional[Sentiment] = None
    should_escalate: bool = False
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Intent(BaseModel):
    intent_id: str
    name: str
    description: str
    confidence_threshold: float = 0.7
    handler: Optional[str] = None
    parameters: Optional[List[str]] = Field(default_factory=list)


class KnowledgeArticle(BaseModel):
    article_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    category: str
    tags: Optional[List[str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    limit: int = 5


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    confidence: float
    query: str

