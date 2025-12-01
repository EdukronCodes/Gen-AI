"""
Ticket Service
Handles ticket CRUD operations
"""
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.ticket import Ticket, TicketStatus, TicketPriority
from datetime import datetime
import uuid


class TicketService:
    """Service for ticket operations"""
    
    def __init__(self):
        pass
    
    def _get_db(self):
        """Get database session"""
        return next(get_db())
    
    async def create_ticket(self, ticket_data: Dict[str, Any]) -> Ticket:
        """Create a new ticket"""
        db = self._get_db()
        try:
            ticket_number = f"TKT-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
            
            ticket = Ticket(
                ticket_number=ticket_number,
                title=ticket_data.get("title", "IT Issue"),
                description=ticket_data.get("description", ""),
                status=TicketStatus.CREATED,
                priority=TicketPriority.P4,
                source_channel=ticket_data.get("source_channel", "web"),
                device_info=ticket_data.get("device_info"),
                location=ticket_data.get("location"),
                impact=ticket_data.get("impact", "medium")
            )
            
            db.add(ticket)
            db.commit()
            db.refresh(ticket)
            
            return ticket
        finally:
            db.close()
    
    async def get_ticket(self, ticket_id: int) -> Optional[Ticket]:
        """Get ticket by ID"""
        db = self._get_db()
        try:
            return db.query(Ticket).filter(Ticket.id == ticket_id).first()
        finally:
            db.close()
    
    async def update_ticket(self, ticket_id: int, update_data: Dict[str, Any]) -> Ticket:
        """Update ticket"""
        db = self._get_db()
        try:
            ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
            if not ticket:
                raise ValueError(f"Ticket {ticket_id} not found")
            
            for key, value in update_data.items():
                if hasattr(ticket, key):
                    setattr(ticket, key, value)
            
            ticket.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(ticket)
            
            return ticket
        finally:
            db.close()
    
    async def assign_ticket(self, ticket_id: int, engineer_id: int) -> Ticket:
        """Assign ticket to engineer"""
        return await self.update_ticket(ticket_id, {
            "assigned_engineer_id": engineer_id,
            "status": TicketStatus.ASSIGNED
        })
    
    async def resolve_ticket(self, ticket_id: int, resolution: str, 
                           auto_resolved: bool = False, resolution_script: str = None) -> Ticket:
        """Resolve ticket"""
        return await self.update_ticket(ticket_id, {
            "status": TicketStatus.RESOLVED,
            "resolution": resolution,
            "auto_resolved": "true" if auto_resolved else "false",
            "resolution_script": resolution_script,
            "resolved_at": datetime.utcnow()
        })
    
    async def close_ticket(self, ticket_id: int) -> Ticket:
        """Close ticket"""
        return await self.update_ticket(ticket_id, {
            "status": TicketStatus.CLOSED,
            "closed_at": datetime.utcnow()
        })

