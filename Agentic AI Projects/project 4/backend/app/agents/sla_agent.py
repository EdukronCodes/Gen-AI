"""
SLA Agent
Predicts urgency and assigns SLA deadlines
"""
from typing import Dict, Any
from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.services.ticket_service import TicketService
from app.core.config import settings


class SLAAgent(BaseAgent):
    """Manages SLA assignment and tracking"""
    
    def __init__(self):
        super().__init__(
            name="SLA Agent",
            role="SLA Manager",
            goal="Assign appropriate SLA and track compliance"
        )
        self.ticket_service = TicketService()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assign priority and SLA"""
        ticket_id = input_data.get("ticket_id")
        ticket = await self.ticket_service.get_ticket(ticket_id)
        
        if not ticket:
            return {"error": "Ticket not found", **input_data}
        
        classification = input_data.get("classification", {})
        severity = classification.get("severity", "medium")
        impact = ticket.impact or "medium"
        
        # Determine priority using LLM
        priority_prompt = f"""
        Determine the priority for this IT ticket:
        
        Title: {ticket.title}
        Description: {ticket.description}
        Category: {ticket.category}
        Severity: {severity}
        Impact: {impact}
        
        Assign priority:
        - P1 (Critical): System down, business critical, immediate impact
        - P2 (High): Major functionality affected, high business impact
        - P3 (Medium): Moderate impact, can wait
        - P4 (Low): Minor issue, low impact
        
        Return JSON with priority and reasoning.
        """
        
        response = await self.llm.ainvoke(priority_prompt)
        priority_data = self._parse_llm_response(response.content)
        
        priority = priority_data.get("priority", "P3")
        
        # Calculate SLA deadline
        sla_hours = {
            "P1": settings.SLA_P1_HOURS,
            "P2": settings.SLA_P2_HOURS,
            "P3": settings.SLA_P3_HOURS,
            "P4": settings.SLA_P4_HOURS
        }.get(priority, settings.SLA_P3_HOURS)
        
        sla_deadline = datetime.utcnow() + timedelta(hours=sla_hours)
        
        # Update ticket
        update_data = {
            "priority": priority,
            "sla_deadline": sla_deadline,
            "sla_breach_risk": "low",
            "status": "classified"
        }
        
        await self.ticket_service.update_ticket(ticket_id, update_data)
        
        self.log_action("SLA assigned", {
            "ticket_id": ticket_id,
            "priority": priority,
            "sla_deadline": sla_deadline.isoformat()
        })
        
        return {
            **input_data,
            "priority": priority,
            "sla_deadline": sla_deadline.isoformat(),
            "sla_hours": sla_hours
        }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM priority response"""
        import json
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
            else:
                json_str = response
            
            data = json.loads(json_str)
            priority = data.get("priority", "P3")
            # Validate priority
            if priority not in ["P1", "P2", "P3", "P4"]:
                priority = "P3"
            
            return {
                "priority": priority,
                "reasoning": data.get("reasoning", "")
            }
        except:
            return {"priority": "P3", "reasoning": "Default priority"}


