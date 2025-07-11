#!/usr/bin/env python3

import sys
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

# Test the Flask application components
from storage import MemStorage
from openai_service import OpenAIService
from models import User, Product, Order, Chat

def test_flask_components():
    print("🚀 Testing Flask Application Components")
    print("=" * 50)
    
    # Test storage
    print("\n📦 Testing Storage Layer...")
    storage = MemStorage()
    
    # Test user operations
    user = storage.get_user(1)
    print(f"✅ User loaded: {user.username} ({user.email})")
    
    # Test product operations
    products = storage.get_products()
    print(f"✅ Products loaded: {len(products)} items")
    
    # Test search
    search_results = storage.search_products("headphones")
    print(f"✅ Search results: {len(search_results)} items found")
    
    # Test orders
    orders = storage.get_orders_by_user(1)
    print(f"✅ Orders loaded: {len(orders)} orders")
    
    # Test OpenAI service
    print("\n🤖 Testing OpenAI Service...")
    openai_service = OpenAIService(api_key=os.getenv('OPENAI_API_KEY'))
    print(f"✅ OpenAI service available: {openai_service.is_available()}")
    
    # Test chat response
    context = {
        'userName': 'John',
        'preferredCategory': 'Electronics',
        'location': 'New York'
    }
    
    response = openai_service.generate_chat_response(
        "Hello, I need help with my order",
        context
    )
    
    print(f"✅ Chat response generated:")
    print(f"   Message: {response['message'][:50]}...")
    print(f"   Intent: {response['intent']}")
    print(f"   Confidence: {response['confidence']}")
    
    # Test recommendations
    recommendations = openai_service.generate_product_recommendations(
        context, 
        products[:3]
    )
    print(f"✅ Recommendations generated: {len(recommendations)} items")
    
    print("\n🎉 All Flask components working correctly!")
    print("=" * 50)
    
    return True

if __name__ == '__main__':
    try:
        test_flask_components()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)