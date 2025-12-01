"""
Reporting Agent
Generates management reports and insights
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.services.reporting_service import ReportingService


class ReportingAgent(BaseAgent):
    """Generates management reports automatically"""
    
    def __init__(self):
        super().__init__(
            name="Reporting Agent",
            role="Business Intelligence Analyst",
            goal="Generate insights and reports for management"
        )
        self.reporting_service = ReportingService()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reports (non-blocking, async)"""
        # This runs asynchronously and doesn't block the workflow
        
        # Generate weekly report
        report = await self.reporting_service.generate_weekly_report()
        
        # Send to management
        await self.reporting_service.send_report(report)
        
        self.log_action("Report generated", {"report_id": report.get("id")})
        
        return {
            **input_data,
            "report_generated": True
        }


