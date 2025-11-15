"""
Base agent class for all specialized agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.models import AgentTask, TaskStatus
from config import settings


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.current_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.learning_enabled = settings.learning_enabled
    
    @abstractmethod
    def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a task"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "current_tasks": self.current_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "status": "active" if self.current_tasks < settings.max_concurrent_tasks else "busy"
        }
    
    def learn(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback"""
        if self.learning_enabled:
            # Implement learning logic
            pass
    
    def collaborate(self, other_agent: 'BaseAgent', task: AgentTask) -> Dict[str, Any]:
        """Collaborate with another agent"""
        # Implement collaboration logic
        return {"status": "collaboration_not_implemented"}
    
    def _update_task_status(self, task: AgentTask, status: TaskStatus, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Update task status"""
        task.status = status
        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.utcnow()
            task.result = result
            self.completed_tasks += 1
            self.current_tasks -= 1
        elif status == TaskStatus.FAILED:
            task.error = error
            task.completed_at = datetime.utcnow()
            self.failed_tasks += 1
            self.current_tasks -= 1
        elif status == TaskStatus.IN_PROGRESS:
            self.current_tasks += 1

