from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
import json
from models import User, Product, Order, Chat, InsertUser, InsertProduct, InsertOrder, InsertChat

class IStorage(ABC):
    """Abstract storage interface"""
    
    @abstractmethod
    def get_user(self, user_id: int) -> Optional[User]:
        pass
    
    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[User]:
        pass
    
    @abstractmethod
    def create_user(self, user_data: Dict[str, Any]) -> User:
        pass
    
    @abstractmethod
    def update_user(self, user_id: int, updates: Dict[str, Any]) -> Optional[User]:
        pass
    
    @abstractmethod
    def get_products(self) -> List[Product]:
        pass
    
    @abstractmethod
    def get_product(self, product_id: int) -> Optional[Product]:
        pass
    
    @abstractmethod
    def search_products(self, query: str) -> List[Product]:
        pass
    
    @abstractmethod
    def get_products_by_category(self, category: str) -> List[Product]:
        pass
    
    @abstractmethod
    def create_product(self, product_data: Dict[str, Any]) -> Product:
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        pass
    
    @abstractmethod
    def get_orders_by_user(self, user_id: int) -> List[Order]:
        pass
    
    @abstractmethod
    def create_order(self, order_data: Dict[str, Any]) -> Order:
        pass
    
    @abstractmethod
    def update_order_status(self, order_id: str, status: str) -> Optional[Order]:
        pass
    
    @abstractmethod
    def get_chat_history(self, user_id: int) -> List[Chat]:
        pass
    
    @abstractmethod
    def create_chat(self, chat_data: Dict[str, Any]) -> Chat:
        pass
    
    @abstractmethod
    def get_recent_chats(self, user_id: int, limit: int) -> List[Chat]:
        pass

class MemStorage(IStorage):
    """In-memory storage implementation"""
    
    def __init__(self):
        self.users: Dict[int, User] = {}
        self.products: Dict[int, Product] = {}
        self.orders: Dict[str, Order] = {}
        self.chats: Dict[int, Chat] = {}
        self.current_user_id = 1
        self.current_product_id = 1
        self.current_order_id = 1
        self.current_chat_id = 1
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with comprehensive mock data"""
        # Create default user
        default_user = User(
            id=1,
            username="johndoe",
            email="john@example.com",
            preferred_category="Electronics",
            location="New York, NY",
            created_at=datetime.now()
        )
        self.users[1] = default_user
        self.current_user_id = 2
        
        # Create sample products
        products_data = [
            {
                "name": "Wireless Bluetooth Headphones",
                "description": "Premium noise-cancelling audio headphones with 30-hour battery life",
                "price": "149.99",
                "category": "Electronics",
                "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=400&h=300",
                "in_stock": True,
                "specifications": ["Bluetooth 5.0", "30-hour battery", "Active noise cancellation"]
            },
            {
                "name": "Multi-Port Charging Station",
                "description": "Fast charging station with 6 USB ports and wireless charging pad",
                "price": "79.99",
                "category": "Electronics",
                "image_url": "https://images.unsplash.com/photo-1583394838336-acd977736f90?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300",
                "in_stock": True,
                "specifications": ["6 USB ports", "Wireless charging", "Quick charge 3.0"]
            },
            {
                "name": "Smart Fitness Watch",
                "description": "Advanced health monitoring with GPS and heart rate tracking",
                "price": "299.99",
                "category": "Electronics",
                "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=400&h=300",
                "in_stock": True,
                "specifications": ["GPS tracking", "Heart rate monitor", "7-day battery"]
            },
            {
                "name": "Gaming Mechanical Keyboard",
                "description": "RGB backlit mechanical keyboard with tactile switches",
                "price": "129.99",
                "category": "Electronics",
                "image_url": "https://images.unsplash.com/photo-1541140532154-b024d705b90a?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300",
                "in_stock": True,
                "specifications": ["RGB backlit", "Mechanical switches", "Anti-ghosting"]
            },
            {
                "name": "Organic Cotton T-Shirt",
                "description": "Comfortable and sustainable organic cotton t-shirt",
                "price": "24.99",
                "category": "Clothing",
                "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300",
                "in_stock": True,
                "specifications": ["100% organic cotton", "Machine washable", "Various sizes"]
            },
            {
                "name": "Ceramic Coffee Mug Set",
                "description": "Set of 4 handcrafted ceramic mugs perfect for coffee lovers",
                "price": "34.99",
                "category": "Home & Kitchen",
                "image_url": "https://images.unsplash.com/photo-1514228742587-6b1558fcf93a?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=300",
                "in_stock": True,
                "specifications": ["Set of 4", "Dishwasher safe", "12 oz capacity"]
            }
        ]
        
        for product_data in products_data:
            self.create_product(product_data)
        
        # Create sample order
        sample_order = {
            "order_id": "ORD-2024-001",
            "user_id": 1,
            "status": "shipped",
            "tracking_number": "1Z999AA123456789",
            "order_date": datetime(2024, 12, 15),
            "estimated_delivery": datetime(2024, 12, 18),
            "total_amount": "149.99",
            "items": ['{"productId": 1, "quantity": 1, "price": "149.99"}']
        }
        self.create_order(sample_order)
    
    def get_user(self, user_id: int) -> Optional[User]:
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def create_user(self, user_data: Dict[str, Any]) -> User:
        user = User(
            id=self.current_user_id,
            username=user_data['username'],
            email=user_data['email'],
            preferred_category=user_data.get('preferred_category'),
            location=user_data.get('location'),
            created_at=datetime.now()
        )
        self.users[self.current_user_id] = user
        self.current_user_id += 1
        return user
    
    def update_user(self, user_id: int, updates: Dict[str, Any]) -> Optional[User]:
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        return user
    
    def get_products(self) -> List[Product]:
        return list(self.products.values())
    
    def get_product(self, product_id: int) -> Optional[Product]:
        return self.products.get(product_id)
    
    def search_products(self, query: str) -> List[Product]:
        query_lower = query.lower()
        results = []
        for product in self.products.values():
            if (query_lower in product.name.lower() or 
                query_lower in product.description.lower() or
                query_lower in product.category.lower()):
                results.append(product)
        return results
    
    def get_products_by_category(self, category: str) -> List[Product]:
        return [product for product in self.products.values() 
                if product.category.lower() == category.lower()]
    
    def create_product(self, product_data: Dict[str, Any]) -> Product:
        product = Product(
            id=self.current_product_id,
            name=product_data['name'],
            description=product_data['description'],
            price=product_data['price'],
            category=product_data['category'],
            image_url=product_data['image_url'],
            in_stock=product_data.get('in_stock', True),
            specifications=product_data.get('specifications', []),
            created_at=datetime.now()
        )
        self.products[self.current_product_id] = product
        self.current_product_id += 1
        return product
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
    
    def get_orders_by_user(self, user_id: int) -> List[Order]:
        return [order for order in self.orders.values() if order.user_id == user_id]
    
    def create_order(self, order_data: Dict[str, Any]) -> Order:
        order = Order(
            id=self.current_order_id,
            order_id=order_data['order_id'],
            user_id=order_data['user_id'],
            status=order_data['status'],
            tracking_number=order_data.get('tracking_number'),
            order_date=order_data.get('order_date', datetime.now()),
            estimated_delivery=order_data.get('estimated_delivery'),
            total_amount=order_data.get('total_amount', '0.00'),
            items=order_data.get('items', [])
        )
        self.orders[order_data['order_id']] = order
        self.current_order_id += 1
        return order
    
    def update_order_status(self, order_id: str, status: str) -> Optional[Order]:
        order = self.orders.get(order_id)
        if order:
            order.status = status
        return order
    
    def get_chat_history(self, user_id: int) -> List[Chat]:
        return [chat for chat in self.chats.values() if chat.user_id == user_id]
    
    def create_chat(self, chat_data: Dict[str, Any]) -> Chat:
        chat = Chat(
            id=self.current_chat_id,
            user_id=chat_data['user_id'],
            message=chat_data['message'],
            response=chat_data['response'],
            intent=chat_data['intent'],
            confidence=chat_data['confidence'],
            timestamp=datetime.now()
        )
        self.chats[self.current_chat_id] = chat
        self.current_chat_id += 1
        return chat
    
    def get_recent_chats(self, user_id: int, limit: int) -> List[Chat]:
        user_chats = [chat for chat in self.chats.values() if chat.user_id == user_id]
        user_chats.sort(key=lambda x: x.timestamp, reverse=True)
        return user_chats[:limit]