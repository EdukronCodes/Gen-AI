"""
Main FastAPI application for Autonomous Business Intelligence System
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import redis

from config import settings
from app.database import get_db, init_db, get_redis
from app.models import (
    QueryRequest,
    QueryResponse,
    AgentTask,
    TaskType,
    Insight,
    Report,
    DataSource,
    AgentStatus
)
from app.agents.orchestrator_agent import OrchestratorAgent
from app.insight_generator import InsightGenerator

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Autonomous Business Intelligence System API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
orchestrator = OrchestratorAgent()
insight_generator = InsightGenerator()


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Autonomous Business Intelligence System API",
        "version": settings.app_version,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "bi-api"}


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Process natural language query"""
    try:
        # Process query using orchestrator
        result = orchestrator.process_natural_language_query(request.query)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Generate insights if applicable
        insights = []
        if request.format == "json":
            # Try to generate insights from result
            try:
                insight = insight_generator.generate_insight(
                    analysis_result=result,
                    insight_type="trend",  # Determine from result
                    category="general"
                )
                insights.append(insight.dict())
            except Exception as e:
                print(f"Error generating insight: {e}")
        
        return QueryResponse(
            response=result,
            insights=insights,
            status="completed"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/v1/query/async")
async def process_query_async(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Process query asynchronously"""
    # Create task
    task = AgentTask(
        agent_id=orchestrator.agent_id,
        task_type=TaskType.ANALYSIS,
        parameters={"query": request.query}
    )
    
    # Execute in background
    background_tasks.add_task(orchestrator.execute, task)
    
    return {
        "task_id": task.task_id,
        "status": "pending",
        "message": "Query submitted for processing"
    }


@app.get("/api/v1/query/{task_id}")
async def get_query_result(task_id: str, db: Session = Depends(get_db)):
    """Get query result by task ID"""
    # In production, retrieve from database
    return {
        "task_id": task_id,
        "status": "completed",
        "message": "Retrieve task from database in production"
    }


@app.get("/api/v1/agents", response_model=List[AgentStatus])
async def list_agents():
    """List all agents and their status"""
    agents_status = []
    
    # Get orchestrator status
    agents_status.append(AgentStatus(**orchestrator.get_status()))
    
    # Get specialized agent statuses
    for agent_id, agent in orchestrator.agents.items():
        agents_status.append(AgentStatus(**agent.get_status()))
    
    return agents_status


@app.get("/api/v1/agents/{agent_id}/status", response_model=AgentStatus)
async def get_agent_status(agent_id: str):
    """Get specific agent status"""
    if agent_id == orchestrator.agent_id:
        return AgentStatus(**orchestrator.get_status())
    
    if agent_id in orchestrator.agents:
        return AgentStatus(**orchestrator.agents[agent_id].get_status())
    
    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/api/v1/agents/{agent_id}/task")
async def assign_task(
    agent_id: str,
    task: AgentTask,
    db: Session = Depends(get_db)
):
    """Assign task to specific agent"""
    if agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = orchestrator.agents[agent_id]
    result = agent.execute(task)
    
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "result": result
    }


@app.get("/api/v1/insights", response_model=List[Insight])
async def get_insights(
    category: Optional[str] = None,
    date_range: Optional[str] = None,
    priority: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get generated insights"""
    # In production, retrieve from database
    from app.database import InsightDB
    
    query = db.query(InsightDB)
    
    if category:
        query = query.filter(InsightDB.category == category)
    if priority:
        query = query.filter(InsightDB.severity == priority)
    
    insights = query.limit(100).all()
    
    return [
        Insight(
            insight_id=insight.insight_id,
            type=insight.type,
            title=insight.title,
            description=insight.description,
            category=insight.category,
            severity=insight.severity,
            confidence=insight.confidence,
            data_points=insight.data_points or [],
            recommendations=insight.recommendations or [],
            business_impact=insight.business_impact,
            created_at=insight.created_at,
            status=insight.status
        )
        for insight in insights
    ]


@app.post("/api/v1/insights/{insight_id}/feedback")
async def provide_feedback(
    insight_id: str,
    rating: int,
    comment: Optional[str] = None,
    action_taken: bool = False,
    db: Session = Depends(get_db)
):
    """Provide feedback on insight"""
    # In production, save feedback to database
    return {
        "insight_id": insight_id,
        "message": "Feedback recorded",
        "rating": rating
    }


@app.get("/api/v1/reports", response_model=List[Report])
async def list_reports(db: Session = Depends(get_db)):
    """List available reports"""
    from app.database import ReportDB
    
    reports = db.query(ReportDB).limit(50).all()
    
    return [
        Report(
            report_id=report.report_id,
            name=report.name,
            type=report.type,
            template=report.template,
            parameters=report.parameters or {},
            format=report.format,
            schedule=report.schedule,
            recipients=report.recipients or [],
            generated_at=report.generated_at,
            download_url=report.download_url,
            metadata=report.metadata or {}
        )
        for report in reports
    ]


@app.post("/api/v1/reports/generate")
async def generate_report(
    name: str,
    template: Optional[str] = None,
    parameters: Optional[dict] = None,
    format: str = "pdf",
    db: Session = Depends(get_db)
):
    """Generate custom report"""
    # In production, implement report generation
    report_id = "generated_report_id"
    
    return {
        "report_id": report_id,
        "download_url": f"/api/v1/reports/{report_id}",
        "message": "Report generation initiated"
    }


@app.get("/api/v1/data-sources", response_model=List[DataSource])
async def list_data_sources(db: Session = Depends(get_db)):
    """List connected data sources"""
    from app.database import DataSourceDB
    
    sources = db.query(DataSourceDB).all()
    
    return [
        DataSource(
            source_id=source.source_id,
            name=source.name,
            type=source.type,
            connection=source.connection,
            schema=source.schema,
            status=source.status,
            last_refresh=source.last_refresh,
            metadata=source.metadata or {}
        )
        for source in sources
    ]


@app.post("/api/v1/data-sources", response_model=DataSource)
async def add_data_source(
    name: str,
    type: str,
    connection: dict,
    schema: Optional[dict] = None,
    db: Session = Depends(get_db)
):
    """Add new data source"""
    from app.database import DataSourceDB
    import uuid
    
    source = DataSourceDB(
        source_id=str(uuid.uuid4()),
        name=name,
        type=type,
        connection=connection,
        schema=schema,
        status="active"
    )
    
    db.add(source)
    db.commit()
    db.refresh(source)
    
    return DataSource(
        source_id=source.source_id,
        name=source.name,
        type=source.type,
        connection=source.connection,
        schema=source.schema,
        status=source.status,
        last_refresh=source.last_refresh,
        metadata=source.metadata or {}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)

