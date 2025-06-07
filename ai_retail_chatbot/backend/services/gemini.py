import requests

class GeminiService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'

    def get_chat_response(self, user_message):
        headers = {
            'Content-Type': 'application/json',
        }
        params = {
            'key': self.api_key
        }
        data = {
            "contents": [
                {"parts": [{"text": user_message}]}
            ]
        }
        try:
            response = requests.post(self.api_url, headers=headers, params=params, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            # Extract the generated text from Gemini API response
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return "Sorry, I couldn't process your request right now." 