"""
Engineer Service
Manages engineer data and workload
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.engineer import Engineer


class EngineerService:
    """Service for engineer operations"""
    
    def __init__(self):
        pass
    
    def _get_db(self):
        """Get database session"""
        return next(get_db())
    
    async def get_available_engineers(self) -> List[Engineer]:
        """Get all available engineers"""
        db = self._get_db()
        try:
            return db.query(Engineer).filter(
                Engineer.is_available == "true"
            ).all()
        finally:
            db.close()
    
    async def get_engineer(self, engineer_id: int) -> Optional[Engineer]:
        """Get engineer by ID"""
        db = self._get_db()
        try:
            return db.query(Engineer).filter(Engineer.id == engineer_id).first()
        finally:
            db.close()
    
    async def increment_workload(self, engineer_id: int):
        """Increment engineer's active ticket count"""
        db = self._get_db()
        try:
            engineer = db.query(Engineer).filter(Engineer.id == engineer_id).first()
            if engineer:
                engineer.active_tickets_count += 1
                db.commit()
        finally:
            db.close()
    
    async def decrement_workload(self, engineer_id: int):
        """Decrement engineer's active ticket count"""
        db = self._get_db()
        try:
            engineer = db.query(Engineer).filter(Engineer.id == engineer_id).first()
            if engineer:
                engineer.active_tickets_count = max(0, engineer.active_tickets_count - 1)
                db.commit()
        finally:
            db.close()

