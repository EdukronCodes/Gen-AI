EMOTIONS = {
    "anger": ["angry", "furious", "mad", "outraged"],
    "frustration": ["frustrated", "annoyed", "irritated"],
    "happiness": ["happy", "glad", "pleased", "delighted"],
    "sadness": ["sad", "disappointed", "upset"],
}


def detect_emotion(text: str) -> str | None:
    lower = text.lower()
    for emotion, keywords in EMOTIONS.items():
        if any(kw in lower for kw in keywords):
            return emotion
    return None
