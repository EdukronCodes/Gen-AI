from sqlalchemy.orm import Session
from app.agents.orchestrator.agent_router import AgentType, detect_intent, extract_order_number
from app.agents.orchestrator.decision_manager import DecisionManager
from app.agents.customer_support_agent.agent import CustomerSupportAgent
from app.agents.order_tracking_agent.agent import OrderTrackingAgent
from app.agents.refund_agent.agent import RefundAgent
from app.agents.recommendation_agent.agent import RecommendationAgent
from app.agents.escalation_agent.agent import EscalationAgent
from app.agents.sentiment_agent.classifier import SentimentClassifier


class WorkflowEngine:
    def __init__(self, db: Session):
        self.db = db
        self.sentiment = SentimentClassifier()
        self.decision = DecisionManager()
        self.agents = {
            AgentType.CUSTOMER_SUPPORT: CustomerSupportAgent(db),
            AgentType.ORDER_TRACKING: OrderTrackingAgent(db),
            AgentType.REFUND: RefundAgent(db),
            AgentType.RECOMMENDATION: RecommendationAgent(db),
            AgentType.ESCALATION: EscalationAgent(db),
        }

    def process(self, message: str, customer_email: str | None = None, session_id: str | None = None) -> dict:
        sentiment = self.sentiment.classify(message)
        if self.decision.should_escalate(message, sentiment["score"]):
            agent_type = AgentType.ESCALATION
        else:
            agent_type = detect_intent(message)

        agent = self.agents[agent_type]
        context = {
            "message": message,
            "customer_email": customer_email,
            "session_id": session_id,
            "sentiment": sentiment,
            "order_number": extract_order_number(message),
        }
        response = agent.handle(context)
        return {
            "reply": response["reply"],
            "agent": agent_type.value,
            "sentiment": sentiment,
            "metadata": response.get("metadata", {}),
            "escalated": agent_type == AgentType.ESCALATION,
        }
