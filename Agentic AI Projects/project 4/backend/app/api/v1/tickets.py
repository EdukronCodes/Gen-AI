"""
Ticket API Endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
from pydantic import BaseModel
from app.agents.orchestrator_agent import OrchestratorAgent
from app.services.ticket_service import TicketService
from app.models.ticket import Ticket

router = APIRouter()
orchestrator = OrchestratorAgent()
ticket_service = TicketService()


class TicketCreateRequest(BaseModel):
    title: str
    description: str
    channel: str = "web"
    device_info: dict = None
    location: str = None
    impact: str = "medium"


class TicketResponse(BaseModel):
    id: int
    ticket_number: str
    title: str
    description: str
    status: str
    priority: str
    category: str = None
    
    class Config:
        from_attributes = True


@router.post("/", response_model=TicketResponse)
async def create_ticket(request: TicketCreateRequest):
    """Create a new ticket and start agentic workflow"""
    try:
        # Start orchestrator workflow
        result = await orchestrator.process({
            "stage": "intake",
            "channel": request.channel,
            "raw_input": request.description,
            "title": request.title,
            "device_info": request.device_info,
            "location": request.location,
            "impact": request.impact
        })
        
        ticket_id = result.get("ticket_id")
        ticket = await ticket_service.get_ticket(ticket_id)
        
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        return ticket
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: int):
    """Get ticket by ID"""
    ticket = await ticket_service.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.get("/", response_model=List[TicketResponse])
async def list_tickets(limit: int = 100, offset: int = 0):
    """List all tickets"""
    # In production, implement proper pagination
    from app.core.database import get_db
    db = next(get_db())
    tickets = db.query(Ticket).offset(offset).limit(limit).all()
    return tickets


