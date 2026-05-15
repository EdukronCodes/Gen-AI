import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from app.database.models import Ticket
from app.agents.escalation_agent.severity_classifier import classify_severity


class EscalationAgent:
    def __init__(self, db: Session):
        self.db = db

    def handle(self, context: dict) -> dict:
        message = context["message"]
        sentiment = context.get("sentiment", {})
        priority = classify_severity(sentiment.get("score", 0))

        ticket_num = f"TKT-{uuid.uuid4().hex[:6].upper()}"
        ticket = Ticket(
            ticket_number=ticket_num,
            subject="Escalated from chatbot",
            description=message[:500],
            status="open",
            priority=priority,
        )
        self.db.add(ticket)
        self.db.commit()

        return {
            "reply": (
                f"I understand your frustration and want to make this right. "
                f"I've created support ticket **{ticket_num}** with **{priority}** priority.\n\n"
                f"A human agent will contact you within "
                f"{'2 hours' if priority == 'high' else '24 hours'}. "
                f"Is there anything else I can note for the agent?"
            ),
            "metadata": {"ticket_number": ticket_num, "priority": priority},
        }
