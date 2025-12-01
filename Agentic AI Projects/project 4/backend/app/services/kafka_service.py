"""
Kafka Service
Handles event streaming
"""
from typing import Dict, Any
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
from app.core.config import settings


class KafkaService:
    """Service for Kafka event streaming"""
    
    def __init__(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS.split(","),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
        except Exception as e:
            print(f"Kafka connection error: {e}")
            self.producer = None
    
    async def publish_ticket_event(self, event: Dict[str, Any]):
        """Publish ticket event to Kafka"""
        if not self.producer:
            print("Kafka producer not available, skipping event")
            return
        
        try:
            topic = settings.KAFKA_TICKET_TOPIC
            key = event.get("ticket_id") or event.get("ticket_number", "unknown")
            
            future = self.producer.send(topic, key=str(key), value=event)
            future.get(timeout=10)  # Wait for send confirmation
            
        except KafkaError as e:
            print(f"Kafka publish error: {e}")
    
    async def publish_general_event(self, event: Dict[str, Any]):
        """Publish general event to Kafka"""
        if not self.producer:
            print("Kafka producer not available, skipping event")
            return
        
        try:
            topic = settings.KAFKA_EVENTS_TOPIC
            future = self.producer.send(topic, value=event)
            future.get(timeout=10)
            
        except KafkaError as e:
            print(f"Kafka publish error: {e}")


