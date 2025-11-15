"""
Conversation management and persistence
"""
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime
from app.database import ConversationDB, MessageDB
from app.models import Conversation, Message, ConversationStatus
import uuid


class ConversationManager:
    """Manage conversations and messages"""
    
    def create_conversation(
        self,
        db: Session,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        channel: str = "web"
    ) -> Conversation:
        """Create a new conversation"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        db_conversation = ConversationDB(
            conversation_id=conversation_id,
            user_id=user_id,
            channel=channel,
            status="active"
        )
        
        db.add(db_conversation)
        db.commit()
        db.refresh(db_conversation)
        
        return Conversation(
            conversation_id=db_conversation.conversation_id,
            user_id=db_conversation.user_id,
            channel=db_conversation.channel,
            status=ConversationStatus(db_conversation.status),
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at,
            metadata=db_conversation.metadata or {}
        )
    
    def get_conversation(self, db: Session, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        db_conversation = db.query(ConversationDB).filter(
            ConversationDB.conversation_id == conversation_id
        ).first()
        
        if not db_conversation:
            return None
        
        return Conversation(
            conversation_id=db_conversation.conversation_id,
            user_id=db_conversation.user_id,
            channel=db_conversation.channel,
            status=ConversationStatus(db_conversation.status),
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at,
            metadata=db_conversation.metadata or {}
        )
    
    def get_conversation_history(self, db: Session, conversation_id: str) -> List[Dict]:
        """Get conversation message history"""
        messages = db.query(MessageDB).filter(
            MessageDB.conversation_id == conversation_id
        ).order_by(MessageDB.timestamp).all()
        
        return [
            {
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata or {}
            }
            for msg in messages
        ]
    
    def save_message(
        self,
        db: Session,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Message:
        """Save message to database"""
        message_id = str(uuid.uuid4())
        
        db_message = MessageDB(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        
        return Message(
            message_id=db_message.message_id,
            conversation_id=db_message.conversation_id,
            role=role,
            content=db_message.content,
            timestamp=db_message.timestamp,
            metadata=db_message.metadata or {}
        )
    
    def update_conversation_status(
        self,
        db: Session,
        conversation_id: str,
        status: str
    ) -> bool:
        """Update conversation status"""
        db_conversation = db.query(ConversationDB).filter(
            ConversationDB.conversation_id == conversation_id
        ).first()
        
        if not db_conversation:
            return False
        
        db_conversation.status = status
        db_conversation.updated_at = datetime.utcnow()
        db.commit()
        
        return True
    
    def update_conversation_timestamp(self, db: Session, conversation_id: str):
        """Update conversation timestamp"""
        db_conversation = db.query(ConversationDB).filter(
            ConversationDB.conversation_id == conversation_id
        ).first()
        
        if db_conversation:
            db_conversation.updated_at = datetime.utcnow()
            db.commit()

