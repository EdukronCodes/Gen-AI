import uuid
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.config.database import get_db
from app.agents.orchestrator.workflow_engine import WorkflowEngine

router = APIRouter(prefix="/api/v1/customer", tags=["Customer"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    customer_email: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    agent: str
    sentiment: dict
    session_id: str
    escalated: bool = False
    metadata: dict = {}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    session_id = request.session_id or str(uuid.uuid4())
    engine = WorkflowEngine(db)
    result = engine.process(request.message, request.customer_email, session_id)
    return ChatResponse(
        reply=result["reply"],
        agent=result["agent"],
        sentiment=result["sentiment"],
        session_id=session_id,
        escalated=result.get("escalated", False),
        metadata=result.get("metadata", {}),
    )
