from app.agents.sentiment_agent.classifier import SentimentClassifier


class DecisionManager:
    def __init__(self):
        self.sentiment = SentimentClassifier()

    def should_escalate(self, message: str, sentiment_score: float) -> bool:
        if sentiment_score < -0.6:
            return True
        escalation_keywords = ["lawyer", "sue", "never again", "worst", "scam", "fraud"]
        return any(kw in message.lower() for kw in escalation_keywords)

    def get_priority(self, sentiment_score: float) -> str:
        if sentiment_score < -0.5:
            return "high"
        if sentiment_score < -0.2:
            return "medium"
        return "low"
