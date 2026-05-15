import re

NEGATIVE = ["angry", "frustrated", "terrible", "awful", "hate", "worst", "broken", "damaged", "late", "never", "scam"]
POSITIVE = ["thank", "great", "awesome", "love", "happy", "excellent", "perfect", "wonderful"]


class SentimentClassifier:
    def classify(self, text: str) -> dict:
        lower = text.lower()
        neg = sum(1 for w in NEGATIVE if w in lower)
        pos = sum(1 for w in POSITIVE if w in lower)
        if neg > pos:
            score = max(-1.0, -0.3 * neg)
            label = "negative"
        elif pos > neg:
            score = min(1.0, 0.3 * pos)
            label = "positive"
        else:
            score = 0.0
            label = "neutral"
        if re.search(r"!{2,}", text):
            score -= 0.2 if label == "negative" else 0.1
        return {"label": label, "score": round(score, 2)}
