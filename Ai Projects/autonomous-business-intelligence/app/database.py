"""
Database connection and session management
"""
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text, Integer, Float, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from typing import Generator
import redis
from pymongo import MongoClient
from config import settings

# PostgreSQL Database
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis for caching
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

# MongoDB for analytics
mongo_client = MongoClient(settings.mongodb_url)
mongo_db = mongo_client.get_database("bi_analytics")


class AgentTaskDB(Base):
    __tablename__ = "agent_tasks"
    
    task_id = Column(String, primary_key=True)
    agent_id = Column(String, nullable=False, index=True)
    task_type = Column(String, nullable=False)
    parameters = Column(JSON, default={})
    priority = Column(String, default="medium")
    status = Column(String, default="pending")
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)


class InsightDB(Base):
    __tablename__ = "insights"
    
    insight_id = Column(String, primary_key=True)
    type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    data_points = Column(JSON, default=[])
    visualizations = Column(JSON, nullable=True)
    recommendations = Column(JSON, default=[])
    business_impact = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    status = Column(String, default="new")


class ReportDB(Base):
    __tablename__ = "reports"
    
    report_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    template = Column(Text, nullable=True)
    parameters = Column(JSON, default={})
    format = Column(String, default="pdf")
    schedule = Column(JSON, nullable=True)
    recipients = Column(JSON, default=[])
    generated_at = Column(DateTime, nullable=True)
    download_url = Column(String, nullable=True)
    metadata = Column(JSON, default={})


class DataSourceDB(Base):
    __tablename__ = "data_sources"
    
    source_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    connection = Column(JSON, nullable=False)
    schema = Column(JSON, nullable=True)
    status = Column(String, default="active")
    last_refresh = Column(DateTime, nullable=True)
    metadata = Column(JSON, default={})


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Get Redis client"""
    return redis_client


def get_mongo() -> MongoClient:
    """Get MongoDB client"""
    return mongo_db

