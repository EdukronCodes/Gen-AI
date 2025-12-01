"""
Monitoring Agent
Monitors infrastructure and auto-creates tickets
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.services.monitoring_service import MonitoringService
from app.agents.orchestrator_agent import OrchestratorAgent


class MonitoringAgent(BaseAgent):
    """Monitors infrastructure and creates tickets automatically"""
    
    def __init__(self):
        super().__init__(
            name="Monitoring Agent",
            role="Infrastructure Monitor",
            goal="Detect infrastructure issues and create tickets automatically"
        )
        self.monitoring_service = MonitoringService()
        self.orchestrator = OrchestratorAgent()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor infrastructure (called periodically)"""
        # This would typically run on a schedule
        # For now, we'll just log that monitoring is active
        
        # Check for infrastructure alerts
        alerts = await self.monitoring_service.get_active_alerts()
        
        for alert in alerts:
            # Auto-create ticket for critical alerts
            if alert.get("severity") in ["critical", "high"]:
                ticket_data = {
                    "channel": "monitoring",
                    "raw_input": f"Infrastructure Alert: {alert.get('message')}",
                    "title": alert.get("title", "Infrastructure Issue"),
                    "device_info": alert.get("device_info"),
                    "impact": alert.get("severity")
                }
                
                # Create ticket via orchestrator
                result = await self.orchestrator.process({
                    "stage": "intake",
                    **ticket_data
                })
                
                self.log_action("Auto-created ticket from monitoring", {
                    "alert_id": alert.get("id"),
                    "ticket_id": result.get("ticket_id")
                })
        
        return {
            **input_data,
            "monitoring_status": "active",
            "alerts_checked": len(alerts)
        }


