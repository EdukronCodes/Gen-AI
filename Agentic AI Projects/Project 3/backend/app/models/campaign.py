"""
Campaign models for managing social media campaigns
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class CampaignStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Campaign(Base):
    __tablename__ = "campaigns"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    goal = Column(Text, nullable=False)  # User's goal/objective
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT)
    target_platforms = Column(JSON, default=list)  # ["instagram", "twitter", "facebook", "youtube"]
    duration_days = Column(Integer, default=7)
    content_themes = Column(JSON, default=list)
    target_audience = Column(JSON, default=dict)
    strategy_output = Column(JSON, nullable=True)  # Strategy agent output
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    posts = relationship("CampaignPost", back_populates="campaign", cascade="all, delete-orphan")


class CampaignPost(Base):
    __tablename__ = "campaign_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=False)
    platform = Column(String, nullable=False)  # instagram, twitter, facebook, youtube
    content_type = Column(String, nullable=False)  # post, reel, story, thread, video, shorts
    content = Column(JSON, nullable=False)  # Generated content (caption, script, etc.)
    media_paths = Column(JSON, default=list)  # Paths to generated media
    scheduled_time = Column(DateTime(timezone=True), nullable=True)
    posted_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, default="pending")  # pending, scheduled, posted, failed
    platform_post_id = Column(String, nullable=True)  # ID from platform API
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    campaign = relationship("Campaign", back_populates="posts")

