"""
RCA Agent
Generates root cause analysis automatically
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.services.ticket_service import TicketService
from app.services.log_service import LogService


class RCAAgent(BaseAgent):
    """Generates root cause analysis using GenAI"""
    
    def __init__(self):
        super().__init__(
            name="RCA Agent",
            role="Root Cause Analyst",
            goal="Generate comprehensive root cause analysis automatically"
        )
        self.ticket_service = TicketService()
        self.log_service = LogService()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate root cause analysis"""
        ticket_id = input_data.get("ticket_id")
        ticket = await self.ticket_service.get_ticket(ticket_id)
        
        if not ticket:
            return {"error": "Ticket not found", **input_data}
        
        # Gather context
        logs = await self.log_service.get_ticket_logs(ticket_id)
        metrics = await self.log_service.get_ticket_metrics(ticket_id)
        
        # Generate RCA using LLM
        rca_prompt = f"""
        Generate a comprehensive root cause analysis for this IT incident:
        
        Ticket:
        - Title: {ticket.title}
        - Description: {ticket.description}
        - Category: {ticket.category}
        - Root Symptom: {ticket.root_symptom}
        - Resolution: {ticket.resolution or 'N/A'}
        
        System Logs:
        {self._format_logs(logs[:50])}  # Last 50 log entries
        
        Metrics:
        {self._format_metrics(metrics)}
        
        Provide:
        1. Root Cause: Primary cause of the issue
        2. Contributing Factors: Secondary causes
        3. Impact Analysis: What was affected
        4. Resolution Summary: How it was fixed
        5. Preventive Measures: How to prevent recurrence
        
        Return detailed JSON format.
        """
        
        response = await self.llm.ainvoke(rca_prompt)
        rca_data = self._parse_llm_response(response.content)
        
        # Format RCA document
        rca_document = f"""
ROOT CAUSE ANALYSIS
===================

Ticket: {ticket.ticket_number}
Date: {ticket.created_at}

ROOT CAUSE:
{rca_data.get('root_cause', 'N/A')}

CONTRIBUTING FACTORS:
{rca_data.get('contributing_factors', 'N/A')}

IMPACT ANALYSIS:
{rca_data.get('impact_analysis', 'N/A')}

RESOLUTION SUMMARY:
{rca_data.get('resolution_summary', ticket.resolution or 'N/A')}

PREVENTIVE MEASURES:
{rca_data.get('preventive_measures', 'N/A')}
        """
        
        # Update ticket
        await self.ticket_service.update_ticket(ticket_id, {
            "root_cause_analysis": rca_document
        })
        
        # Store in knowledge base for future reference
        # (This would be handled by a separate service)
        
        self.log_action("RCA generated", {"ticket_id": ticket_id})
        
        return {
            **input_data,
            "rca_generated": True,
            "rca_summary": rca_data
        }
    
    def _format_logs(self, logs: list) -> str:
        """Format logs for LLM context"""
        if not logs:
            return "No logs available."
        
        formatted = []
        for log in logs[:20]:  # Limit to 20 most recent
            formatted.append(f"[{log.get('timestamp')}] {log.get('level')}: {log.get('message')}")
        
        return "\n".join(formatted)
    
    def _format_metrics(self, metrics: dict) -> str:
        """Format metrics for LLM context"""
        if not metrics:
            return "No metrics available."
        
        formatted = []
        for key, value in metrics.items():
            formatted.append(f"{key}: {value}")
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM RCA response"""
        import json
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
            else:
                json_str = response
            
            return json.loads(json_str)
        except:
            return {
                "root_cause": "Unable to determine root cause",
                "contributing_factors": "N/A",
                "impact_analysis": "N/A",
                "resolution_summary": "N/A",
                "preventive_measures": "N/A"
            }


