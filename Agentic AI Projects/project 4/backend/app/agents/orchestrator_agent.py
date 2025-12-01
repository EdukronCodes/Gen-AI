"""
Orchestrator Agent
Coordinates all other agents in the workflow
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.agents.intake_agent import IntakeAgent
from app.agents.classification_agent import ClassificationAgent
from app.agents.sla_agent import SLAAgent
from app.agents.assignment_agent import AssignmentAgent
from app.agents.resolution_agent import ResolutionAgent
from app.agents.monitoring_agent import MonitoringAgent
from app.agents.escalation_agent import EscalationAgent
from app.agents.rca_agent import RCAAgent
from app.agents.reporting_agent import ReportingAgent
import asyncio


class OrchestratorAgent(BaseAgent):
    """Orchestrates the entire ticket lifecycle"""
    
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            role="Workflow Coordinator",
            goal="Coordinate all agents to autonomously handle IT tickets"
        )
        
        # Initialize all agents
        self.intake_agent = IntakeAgent()
        self.classification_agent = ClassificationAgent()
        self.sla_agent = SLAAgent()
        self.assignment_agent = AssignmentAgent()
        self.resolution_agent = ResolutionAgent()
        self.monitoring_agent = MonitoringAgent()
        self.escalation_agent = EscalationAgent()
        self.rca_agent = RCAAgent()
        self.reporting_agent = ReportingAgent()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration flow:
        1. Intake -> 2. Classify -> 3. SLA -> 4. Assign -> 5. Resolve -> 6. Monitor -> 7. Escalate -> 8. RCA -> 9. Report
        """
        ticket_id = input_data.get("ticket_id")
        workflow_stage = input_data.get("stage", "intake")
        
        result = {
            "ticket_id": ticket_id,
            "workflow_stage": workflow_stage,
            "status": "in_progress"
        }
        
        try:
            if workflow_stage == "intake":
                # Phase 1: Intake
                result = await self.intake_agent.process(input_data)
                result["workflow_stage"] = "classification"
                
            if workflow_stage == "classification" or result.get("workflow_stage") == "classification":
                # Phase 2: Classification
                result = await self.classification_agent.process(result)
                result["workflow_stage"] = "sla"
            
            if workflow_stage == "sla" or result.get("workflow_stage") == "sla":
                # Phase 3: SLA Assignment
                result = await self.sla_agent.process(result)
                result["workflow_stage"] = "assignment"
            
            if workflow_stage == "assignment" or result.get("workflow_stage") == "assignment":
                # Phase 4: Auto-Assignment
                result = await self.assignment_agent.process(result)
                result["workflow_stage"] = "resolution"
            
            if workflow_stage == "resolution" or result.get("workflow_stage") == "resolution":
                # Phase 5: Auto-Resolution
                result = await self.resolution_agent.process(result)
                if result.get("auto_resolved") == "true":
                    result["workflow_stage"] = "rca"
                else:
                    result["workflow_stage"] = "monitoring"
            
            if workflow_stage == "monitoring" or result.get("workflow_stage") == "monitoring":
                # Phase 6: Continuous Monitoring
                await self.monitoring_agent.process(result)
                result["workflow_stage"] = "escalation_check"
            
            if workflow_stage == "escalation_check" or result.get("workflow_stage") == "escalation_check":
                # Phase 7: Escalation Check
                escalation_result = await self.escalation_agent.process(result)
                if escalation_result.get("escalated"):
                    result.update(escalation_result)
                result["workflow_stage"] = "rca"
            
            if workflow_stage == "rca" or result.get("workflow_stage") == "rca":
                # Phase 8: Root Cause Analysis
                rca_result = await self.rca_agent.process(result)
                result.update(rca_result)
                result["workflow_stage"] = "reporting"
            
            if workflow_stage == "reporting" or result.get("workflow_stage") == "reporting":
                # Phase 9: Reporting (async, non-blocking)
                asyncio.create_task(self.reporting_agent.process(result))
                result["workflow_stage"] = "completed"
            
            result["status"] = "completed"
            self.log_action("Workflow completed", result)
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.log_action("Workflow error", result)
        
        return result


