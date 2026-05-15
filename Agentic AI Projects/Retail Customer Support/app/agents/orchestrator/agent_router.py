import re
from enum import Enum


class AgentType(str, Enum):
    CUSTOMER_SUPPORT = "customer_support"
    ORDER_TRACKING = "order_tracking"
    REFUND = "refund"
    RECOMMENDATION = "recommendation"
    ESCALATION = "escalation"


INTENT_PATTERNS = {
    AgentType.ORDER_TRACKING: [
        r"\b(track|tracking|where is|shipment|shipping|delivery|delivered)\b",
        r"\bORD-\d+\b",
        r"\border\s*(status|number)?\b",
    ],
    AgentType.REFUND: [
        r"\b(refund|return|money back|exchange|damaged|broken)\b",
        r"\bREF-\d+\b",
    ],
    AgentType.RECOMMENDATION: [
        r"\b(recommend|suggest|looking for|show me|browse|product)\b",
        r"\b(headphones|keyboard|shoes|jeans|shirt|electronics|apparel)\b",
    ],
    AgentType.ESCALATION: [
        r"\b(speak to|human|manager|supervisor|complaint|angry|frustrated|terrible)\b",
        r"\b(escalat|ticket|help me now)\b",
    ],
}


def detect_intent(message: str) -> AgentType:
    text = message.lower()
    scores = {agent: 0 for agent in AgentType}
    for agent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                scores[agent] += 1
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return AgentType.CUSTOMER_SUPPORT


def extract_order_number(message: str) -> str | None:
    match = re.search(r"ORD-\d+", message, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_refund_number(message: str) -> str | None:
    match = re.search(r"REF-\d+", message, re.IGNORECASE)
    return match.group(0).upper() if match else None
