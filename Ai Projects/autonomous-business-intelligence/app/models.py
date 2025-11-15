"""
Data models for the BI application
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    REPORT = "report"
    INSIGHT = "insight"
    ANOMALY_DETECTION = "anomaly_detection"
    DATA_COLLECTION = "data_collection"


class InsightType(str, Enum):
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    CAUSAL = "causal"


class InsightSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InsightStatus(str, Enum):
    NEW = "new"
    REVIEWED = "reviewed"
    ACTIONED = "actioned"
    DISMISSED = "dismissed"


class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    task_type: TaskType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: str = "medium"  # high, medium, low
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    format: str = "json"  # json, visualization


class QueryResponse(BaseModel):
    response: Dict[str, Any]
    insights: List[Dict[str, Any]] = Field(default_factory=list)
    visualizations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    task_id: Optional[str] = None
    status: str = "completed"


class Insight(BaseModel):
    insight_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: InsightType
    title: str
    description: str
    category: str
    severity: InsightSeverity
    confidence: float
    data_points: List[Dict[str, Any]] = Field(default_factory=list)
    visualizations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    business_impact: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: InsightStatus = InsightStatus.NEW


class Report(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # executive, operational, financial, custom
    template: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    format: str = "pdf"  # pdf, excel, html, json
    schedule: Optional[Dict[str, Any]] = None
    recipients: List[str] = Field(default_factory=list)
    generated_at: Optional[datetime] = None
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataSource(BaseModel):
    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # database, api, file, cloud
    connection: Dict[str, Any]
    schema: Optional[Dict[str, Any]] = None
    status: str = "active"
    last_refresh: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentStatus(BaseModel):
    agent_id: str
    status: str
    capabilities: List[str]
    current_tasks: int
    completed_tasks: int
    failed_tasks: int
    metrics: Dict[str, Any] = Field(default_factory=dict)

