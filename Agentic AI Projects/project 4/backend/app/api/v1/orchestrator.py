"""
Orchestrator API Endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.agents.orchestrator_agent import OrchestratorAgent

router = APIRouter()
orchestrator = OrchestratorAgent()


class OrchestratorRequest(BaseModel):
    stage: str = "intake"
    ticket_id: int = None
    data: Dict[str, Any] = {}


@router.post("/process")
async def process_workflow(request: OrchestratorRequest):
    """Process workflow stage"""
    try:
        input_data = {
            "stage": request.stage,
            "ticket_id": request.ticket_id,
            **request.data
        }
        
        result = await orchestrator.process(input_data)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


