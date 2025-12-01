"""
Analytics models for tracking performance
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"), nullable=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=True)
    platform = Column(String, nullable=False)
    metric_type = Column(String, nullable=False)  # engagement, reach, impressions, clicks, etc.
    metric_value = Column(Float, nullable=False)
    metric_date = Column(DateTime(timezone=True), nullable=False)
    metadata = Column(JSON, default=dict)  # Additional metrics data
    created_at = Column(DateTime(timezone=True), server_default=func.now())

