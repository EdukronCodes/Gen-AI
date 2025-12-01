"""
Escalation Agent
Tracks SLA and escalates when needed
"""
from typing import Dict, Any
from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.services.ticket_service import TicketService
from app.services.notification_service import NotificationService


class EscalationAgent(BaseAgent):
    """Manages ticket escalation based on SLA"""
    
    def __init__(self):
        super().__init__(
            name="Escalation Agent",
            role="SLA Monitor & Escalator",
            goal="Monitor SLA compliance and escalate when needed"
        )
        self.ticket_service = TicketService()
        self.notification_service = NotificationService()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check SLA and escalate if needed"""
        ticket_id = input_data.get("ticket_id")
        ticket = await self.ticket_service.get_ticket(ticket_id)
        
        if not ticket or not ticket.sla_deadline:
            return {"escalated": False, **input_data}
        
        # Calculate SLA burn rate
        now = datetime.utcnow()
        total_time = (ticket.sla_deadline - ticket.created_at).total_seconds()
        elapsed_time = (now - ticket.created_at).total_seconds()
        burn_rate = (elapsed_time / total_time) * 100 if total_time > 0 else 0
        
        # Determine breach risk
        if now > ticket.sla_deadline:
            breach_risk = "breached"
        elif burn_rate > 80:
            breach_risk = "high"
        elif burn_rate > 60:
            breach_risk = "medium"
        else:
            breach_risk = "low"
        
        # Update breach risk
        await self.ticket_service.update_ticket(ticket_id, {
            "sla_breach_risk": breach_risk
        })
        
        # Escalate if needed
        escalated = False
        if breach_risk in ["high", "breached"]:
            # Escalate to manager
            await self.ticket_service.update_ticket(ticket_id, {
                "status": "escalated"
            })
            
            # Notify manager
            await self.notification_service.notify_escalation(
                ticket_id=ticket_id,
                reason=f"SLA breach risk: {breach_risk}",
                burn_rate=burn_rate
            )
            
            escalated = True
            self.log_action("Ticket escalated", {
                "ticket_id": ticket_id,
                "breach_risk": breach_risk,
                "burn_rate": f"{burn_rate:.1f}%"
            })
        
        return {
            **input_data,
            "escalated": escalated,
            "breach_risk": breach_risk,
            "burn_rate": burn_rate
        }


