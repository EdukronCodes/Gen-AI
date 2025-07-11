import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import logging
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from storage import MemStorage
from openai_service import OpenAIService
from models import Chat, User, Product, Order

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize storage and OpenAI service
storage = MemStorage()
openai_service = OpenAIService(api_key=os.getenv('OPENAI_API_KEY'))

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'database': 'connected',
            'ai': 'connected' if openai_service.is_available() else 'fallback_mode'
        }
    })

# Chat endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('userId', 1)
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get user context
        user = storage.get_user(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get chat history for context
        chat_history = storage.get_recent_chats(user_id, 5)
        
        # Generate AI response
        response = openai_service.generate_chat_response(
            user_message=user_message,
            user_context={
                'userName': user.username,
                'preferredCategory': user.preferred_category,
                'location': user.location,
                'chatHistory': [
                    {'message': chat.message, 'response': chat.response, 'timestamp': chat.timestamp}
                    for chat in chat_history
                ]
            }
        )
        
        # Save chat to storage
        chat_data = {
            'user_id': user_id,
            'message': user_message,
            'response': response['message'],
            'intent': response['intent'],
            'confidence': response['confidence']
        }
        storage.create_chat(chat_data)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Get chat history
@app.route('/api/chat/history/<int:user_id>', methods=['GET'])
def get_chat_history(user_id):
    try:
        chats = storage.get_chat_history(user_id)
        return jsonify([{
            'id': chat.id,
            'message': chat.message,
            'response': chat.response,
            'intent': chat.intent,
            'confidence': chat.confidence,
            'timestamp': chat.timestamp.isoformat()
        } for chat in chats])
    except Exception as e:
        logger.error(f"Chat history error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Get user
@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = storage.get_user(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'preferredCategory': user.preferred_category,
            'location': user.location
        })
    except Exception as e:
        logger.error(f"User fetch error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Get products
@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        products = storage.get_products()
        return jsonify([{
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'price': product.price,
            'category': product.category,
            'imageUrl': product.image_url,
            'inStock': product.in_stock,
            'specifications': product.specifications,
            'createdAt': product.created_at.isoformat()
        } for product in products])
    except Exception as e:
        logger.error(f"Products fetch error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Search products
@app.route('/api/products/search/<query>', methods=['GET'])
def search_products(query):
    try:
        products = storage.search_products(query)
        return jsonify([{
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'price': product.price,
            'category': product.category,
            'imageUrl': product.image_url,
            'inStock': product.in_stock,
            'specifications': product.specifications,
            'createdAt': product.created_at.isoformat()
        } for product in products])
    except Exception as e:
        logger.error(f"Product search error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Get products by category
@app.route('/api/products/category/<category>', methods=['GET'])
def get_products_by_category(category):
    try:
        products = storage.get_products_by_category(category)
        return jsonify([{
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'price': product.price,
            'category': product.category,
            'imageUrl': product.image_url,
            'inStock': product.in_stock,
            'specifications': product.specifications,
            'createdAt': product.created_at.isoformat()
        } for product in products])
    except Exception as e:
        logger.error(f"Category products error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Get orders by user
@app.route('/api/orders/user/<int:user_id>', methods=['GET'])
def get_orders_by_user(user_id):
    try:
        orders = storage.get_orders_by_user(user_id)
        return jsonify([{
            'id': order.id,
            'orderId': order.order_id,
            'userId': order.user_id,
            'status': order.status,
            'trackingNumber': order.tracking_number,
            'orderDate': order.order_date.isoformat(),
            'estimatedDelivery': order.estimated_delivery.isoformat() if order.estimated_delivery else None,
            'totalAmount': order.total_amount,
            'items': order.items
        } for order in orders])
    except Exception as e:
        logger.error(f"Orders fetch error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Get order by ID
@app.route('/api/orders/<order_id>', methods=['GET'])
def get_order(order_id):
    try:
        order = storage.get_order(order_id)
        if not order:
            return jsonify({'error': 'Order not found'}), 404
        
        return jsonify({
            'id': order.id,
            'orderId': order.order_id,
            'userId': order.user_id,
            'status': order.status,
            'trackingNumber': order.tracking_number,
            'orderDate': order.order_date.isoformat(),
            'estimatedDelivery': order.estimated_delivery.isoformat() if order.estimated_delivery else None,
            'totalAmount': order.total_amount,
            'items': order.items
        })
    except Exception as e:
        logger.error(f"Order fetch error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Get recommendations
@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    try:
        user = storage.get_user(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        recommendations = openai_service.generate_product_recommendations(
            user_context={
                'userName': user.username,
                'preferredCategory': user.preferred_category,
                'location': user.location
            },
            available_products=storage.get_products()
        )
        
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        return jsonify([])  # Return empty list on error

# Serve static files in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('NODE_ENV') == 'development')