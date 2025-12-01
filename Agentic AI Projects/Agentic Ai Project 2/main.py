"""
FastAPI Application for Multi-Agent Airlines System
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
import uvicorn
import os

from database.database import get_db, init_db
from orchestrator import AgentOrchestrator

app = FastAPI(
    title="Multi-Agent Airlines System",
    description="Intelligent multi-agent system for airline operations with LlamaIndex integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    orchestrator: str
    routed_to: str
    agent_name: str
    response: str
    success: bool
    data: Optional[Dict[str, Any]] = None


class BookingRequest(BaseModel):
    passenger_email: str
    flight_id: int
    seat_class: str = "economy"
    special_requests: Optional[str] = None


class CheckInRequest(BaseModel):
    booking_reference: str
    seat_number: Optional[str] = None


# Serve static files for UI
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup"""
    try:
        init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"⚠ Database initialization warning: {e}")


@app.get("/ui")
async def serve_ui():
    """Serve the main UI"""
    ui_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(ui_file):
        return FileResponse(ui_file)
    return {"message": "UI not found. Please create static/index.html"}


@app.get("/index.html")
async def serve_index():
    """Redirect root HTML requests to UI"""
    return await serve_ui()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent Airlines System API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "agents": "/api/agents",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "agents": "ready"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


@app.get("/api/agents")
async def get_agents(db: Session = Depends(get_db)):
    """Get information about all available agents"""
    orchestrator = AgentOrchestrator(db)
    return orchestrator.get_agent_info()


@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Process a user query through the multi-agent system
    """
    try:
        orchestrator = AgentOrchestrator(db)
        result = orchestrator.process(request.query, request.context)
        
        return QueryResponse(
            orchestrator=result.get("orchestrator", "AgentOrchestrator"),
            routed_to=result.get("routed_to", "unknown"),
            agent_name=result.get("agent_name", "Unknown Agent"),
            response=result.get("response", "No response generated"),
            success=result.get("success", False),
            data=result.get("data")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/book")
async def create_booking(
    booking: BookingRequest,
    db: Session = Depends(get_db)
):
    """Create a flight booking"""
    try:
        orchestrator = AgentOrchestrator(db)
        context = {
            "passenger_email": booking.passenger_email,
            "flight_id": booking.flight_id,
            "seat_class": booking.seat_class,
            "special_requests": booking.special_requests
        }
        result = orchestrator.process(
            f"Book a {booking.seat_class} class flight",
            context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating booking: {str(e)}")


@app.post("/api/checkin")
async def check_in(
    checkin: CheckInRequest,
    db: Session = Depends(get_db)
):
    """Process flight check-in"""
    try:
        orchestrator = AgentOrchestrator(db)
        context = {
            "booking_reference": checkin.booking_reference,
            "seat_number": checkin.seat_number
        }
        result = orchestrator.process(
            f"Check in for booking {checkin.booking_reference}",
            context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing check-in: {str(e)}")


@app.get("/api/flights")
async def search_flights(
    departure_city: Optional[str] = None,
    arrival_city: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Search for flights"""
    try:
        orchestrator = AgentOrchestrator(db)
        search_agent = orchestrator.agents["flight_search"]
        
        context = {}
        if departure_city:
            context["departure_city"] = departure_city
        if arrival_city:
            context["arrival_city"] = arrival_city
        
        query = f"Find flights from {departure_city or 'anywhere'} to {arrival_city or 'anywhere'}"
        result = search_agent.process(query, context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching flights: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

