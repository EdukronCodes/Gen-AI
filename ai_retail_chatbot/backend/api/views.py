from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from services.gemini import GeminiService

class ChatView(APIView):
    def post(self, request):
        user_message = request.data.get('message')
        if not user_message:
            return Response({'error': 'Message is required.'}, status=status.HTTP_400_BAD_REQUEST)

        gemini = GeminiService(api_key=settings.GEMINI_API_KEY)
        bot_response = gemini.get_chat_response(user_message)
        return Response({'response': bot_response}) 