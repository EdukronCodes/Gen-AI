"""
Log Service
Retrieves logs from ELK, MongoDB, etc.
"""
from typing import List, Dict, Any
from app.core.database import get_mongo_db
from datetime import datetime, timedelta


class LogService:
    """Service for log retrieval"""
    
    def __init__(self):
        self.mongo_db = get_mongo_db()
        self.logs_collection = self.mongo_db["ticket_logs"]
        self.metrics_collection = self.mongo_db["ticket_metrics"]
    
    async def get_ticket_logs(self, ticket_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs for a ticket"""
        logs = self.logs_collection.find(
            {"ticket_id": ticket_id}
        ).sort("timestamp", -1).limit(limit)
        
        return list(logs)
    
    async def get_ticket_metrics(self, ticket_id: int) -> Dict[str, Any]:
        """Get metrics for a ticket"""
        metrics = self.metrics_collection.find_one({"ticket_id": ticket_id})
        return metrics or {}
    
    async def store_log(self, ticket_id: int, level: str, message: str):
        """Store log entry"""
        self.logs_collection.insert_one({
            "ticket_id": ticket_id,
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow()
        })


