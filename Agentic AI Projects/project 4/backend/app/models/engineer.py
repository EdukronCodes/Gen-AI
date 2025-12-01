"""
Engineer Model
"""
from sqlalchemy import Column, Integer, String, DateTime, JSON, Integer as SQLInteger
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class EngineerSkill(Base):
    """Engineer skills model"""
    __tablename__ = "engineer_skills"

    id = Column(Integer, primary_key=True, index=True)
    engineer_id = Column(Integer, ForeignKey("engineers.id"))
    skill_name = Column(String, nullable=False)  # network, server, database, cloud, etc.
    proficiency_level = Column(Integer, default=5)  # 1-10
    certifications = Column(JSON, nullable=True)


class Engineer(Base):
    """Engineer model"""
    __tablename__ = "engineers"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    employee_id = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    
    # Workload
    active_tickets_count = Column(Integer, default=0)
    max_concurrent_tickets = Column(Integer, default=10)
    
    # Performance
    avg_resolution_time_hours = Column(Integer, default=0)
    total_tickets_resolved = Column(Integer, default=0)
    auto_resolution_rate = Column(Integer, default=0)  # percentage
    
    # Availability
    is_available = Column(String, default="true")  # true/false
    current_shift = Column(String, nullable=True)  # day, night, weekend
    
    # Skills
    skills = relationship("EngineerSkill", back_populates="engineer")
    tickets = relationship("Ticket", back_populates="assigned_engineer")
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


EngineerSkill.engineer = relationship("Engineer", back_populates="skills")


