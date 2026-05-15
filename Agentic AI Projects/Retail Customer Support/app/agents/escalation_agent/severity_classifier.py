def classify_severity(sentiment_score: float) -> str:
    if sentiment_score < -0.5:
        return "high"
    if sentiment_score < -0.2:
        return "medium"
    return "low"
