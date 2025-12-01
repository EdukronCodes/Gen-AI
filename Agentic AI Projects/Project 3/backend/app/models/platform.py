"""
Platform connection and post models
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class PlatformConnection(Base):
    __tablename__ = "platform_connections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    platform = Column(String, nullable=False)  # instagram, facebook, twitter, youtube
    platform_user_id = Column(String, nullable=True)  # User ID on the platform
    platform_username = Column(String, nullable=True)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    platform_connection_id = Column(Integer, ForeignKey("platform_connections.id"), nullable=False)
    platform = Column(String, nullable=False)
    platform_post_id = Column(String, unique=True, nullable=True)  # ID from platform API
    content_type = Column(String, nullable=False)
    caption = Column(Text, nullable=True)
    media_urls = Column(JSON, default=list)
    hashtags = Column(JSON, default=list)
    posted_at = Column(DateTime(timezone=True), nullable=True)
    metrics = Column(JSON, default=dict)  # likes, shares, comments, views, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    platform_connection = relationship("PlatformConnection")

