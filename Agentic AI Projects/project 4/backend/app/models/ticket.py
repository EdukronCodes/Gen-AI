"""
Ticket Model
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class TicketStatus(str, enum.Enum):
    """Ticket status enumeration"""
    CREATED = "created"
    CLASSIFIED = "classified"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


class TicketPriority(str, enum.Enum):
    """Ticket priority enumeration"""
    P1 = "P1"  # Critical - 1 hour SLA
    P2 = "P2"  # High - 4 hours SLA
    P3 = "P3"  # Medium - 24 hours SLA
    P4 = "P4"  # Low - 72 hours SLA


class Ticket(Base):
    """Ticket model"""
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    ticket_number = Column(String, unique=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(Enum(TicketStatus), default=TicketStatus.CREATED)
    priority = Column(Enum(TicketPriority), default=TicketPriority.P4)
    
    # Classification
    category = Column(String)  # network, server, app, database
    sub_category = Column(String)
    root_symptom = Column(String)
    
    # Assignment
    assigned_engineer_id = Column(Integer, ForeignKey("engineers.id"), nullable=True)
    assigned_engineer = relationship("Engineer", back_populates="tickets")
    
    # User
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_by = relationship("User", foreign_keys=[created_by_user_id])
    
    # SLA
    sla_deadline = Column(DateTime, nullable=True)
    sla_breach_risk = Column(String, default="low")  # low, medium, high, breached
    
    # Resolution
    resolution = Column(Text, nullable=True)
    resolution_script = Column(String, nullable=True)  # Script executed
    auto_resolved = Column(String, default="false")  # true/false
    
    # RCA
    root_cause_analysis = Column(Text, nullable=True)
    
    # Metadata
    source_channel = Column(String)  # web, chat, email, voice, monitoring
    device_info = Column(JSON, nullable=True)
    location = Column(String, nullable=True)
    impact = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    resolved_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    
    # Agent metadata
    agent_metadata = Column(JSON, nullable=True)  # Store agent decisions


