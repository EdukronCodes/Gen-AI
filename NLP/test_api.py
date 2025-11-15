#!/usr/bin/env python3

import sys
import os
from dotenv import load_dotenv
import json
from flask import Flask
from werkzeug.test import Client
from werkzeug.wrappers import Response

# Load environment variables
load_dotenv()

# Add the server directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from app import app

def test_flask_api():
    print("🧪 Testing Flask API Endpoints")
    print("=" * 50)
    
    # Create test client
    client = app.test_client()
    
    # Test health endpoint
    print("\n💚 Testing Health Check...")
    response = client.get('/api/health')
    assert response.status_code == 200
    health_data = json.loads(response.data)
    print(f"✅ Health check: {health_data['status']}")
    print(f"   AI Service: {health_data['services']['ai']}")
    
    # Test user endpoint
    print("\n👤 Testing User Endpoint...")
    response = client.get('/api/users/1')
    assert response.status_code == 200
    user_data = json.loads(response.data)
    print(f"✅ User loaded: {user_data['username']}")
    
    # Test products endpoint
    print("\n🛍️ Testing Products Endpoint...")
    response = client.get('/api/products')
    assert response.status_code == 200
    products_data = json.loads(response.data)
    print(f"✅ Products loaded: {len(products_data)} items")
    
    # Test product search
    print("\n🔍 Testing Product Search...")
    response = client.get('/api/products/search/headphones')
    assert response.status_code == 200
    search_data = json.loads(response.data)
    print(f"✅ Search results: {len(search_data)} items")
    
    # Test category filter
    print("\n📂 Testing Category Filter...")
    response = client.get('/api/products/category/Electronics')
    assert response.status_code == 200
    category_data = json.loads(response.data)
    print(f"✅ Category products: {len(category_data)} items")
    
    # Test orders endpoint
    print("\n📦 Testing Orders Endpoint...")
    response = client.get('/api/orders/user/1')
    assert response.status_code == 200
    orders_data = json.loads(response.data)
    print(f"✅ Orders loaded: {len(orders_data)} orders")
    
    # Test chat endpoint
    print("\n💬 Testing Chat Endpoint...")
    chat_data = {
        'message': 'Hello, I need help with my order',
        'userId': 1
    }
    response = client.post('/api/chat', 
                          data=json.dumps(chat_data),
                          content_type='application/json')
    assert response.status_code == 200
    chat_response = json.loads(response.data)
    print(f"✅ Chat response:")
    print(f"   Intent: {chat_response['intent']}")
    print(f"   Confidence: {chat_response['confidence']}")
    print(f"   Message: {chat_response['message'][:50]}...")
    
    # Test recommendations
    print("\n🎯 Testing Recommendations...")
    response = client.get('/api/recommendations/1')
    assert response.status_code == 200
    recommendations = json.loads(response.data)
    print(f"✅ Recommendations: {len(recommendations)} items")
    
    print("\n🎉 All Flask API endpoints working correctly!")
    print("=" * 50)
    
    return True

if __name__ == '__main__':
    try:
        test_flask_api()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)