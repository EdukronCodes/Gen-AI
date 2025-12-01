"""
Knowledge Base Model
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class KnowledgeBaseEntry(Base):
    """Knowledge base entry model"""
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    sub_category = Column(String, nullable=True)
    
    # Resolution
    resolution_steps = Column(JSON, nullable=True)  # Array of steps
    resolution_script = Column(String, nullable=True)  # Auto-fix script
    success_rate = Column(Integer, default=0)  # 0-100
    
    # Vector embedding will be stored in ChromaDB
    vector_id = Column(String, nullable=True)
    
    # Metadata
    tags = Column(JSON, nullable=True)
    created_by = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


