from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

@dataclass
class User:
    id: int
    username: str
    email: str
    preferred_category: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime = None

@dataclass
class Product:
    id: int
    name: str
    description: str
    price: str
    category: str
    image_url: str
    in_stock: bool
    specifications: List[str]
    created_at: datetime = None

@dataclass
class Order:
    id: int
    order_id: str
    user_id: int
    status: str
    tracking_number: Optional[str] = None
    order_date: datetime = None
    estimated_delivery: Optional[datetime] = None
    total_amount: str = "0.00"
    items: List[str] = None

@dataclass
class Chat:
    id: int
    user_id: int
    message: str
    response: str
    intent: str
    confidence: float
    timestamp: datetime = None

# Insert schemas (equivalent to Zod schemas)
@dataclass
class InsertUser:
    username: str
    email: str
    preferred_category: Optional[str] = None
    location: Optional[str] = None

@dataclass
class InsertProduct:
    name: str
    description: str
    price: str
    category: str
    image_url: str
    in_stock: bool = True
    specifications: List[str] = None

@dataclass
class InsertOrder:
    order_id: str
    user_id: int
    status: str
    tracking_number: Optional[str] = None
    order_date: datetime = None
    estimated_delivery: Optional[datetime] = None
    total_amount: str = "0.00"
    items: List[str] = None

@dataclass
class InsertChat:
    user_id: int
    message: str
    response: str
    intent: str
    confidence: float