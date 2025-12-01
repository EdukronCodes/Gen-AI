"""
Knowledge Base Service
Manages knowledge base entries and resolution scripts
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.knowledge_base import KnowledgeBaseEntry


class KnowledgeBaseService:
    """Service for knowledge base operations"""
    
    def __init__(self):
        pass
    
    def _get_db(self):
        """Get database session"""
        return next(get_db())
    
    async def find_resolution_scripts(self, category: str = None, 
                                     sub_category: str = None,
                                     root_symptom: str = None) -> List[Dict[str, Any]]:
        """Find resolution scripts matching criteria"""
        db = self._get_db()
        try:
            query = db.query(KnowledgeBaseEntry).filter(
                KnowledgeBaseEntry.resolution_script.isnot(None)
            )
            
            if category:
                query = query.filter(KnowledgeBaseEntry.category == category)
            
            if sub_category:
                query = query.filter(KnowledgeBaseEntry.sub_category == sub_category)
            
            entries = query.all()
            
            results = []
            for entry in entries:
                results.append({
                    "id": entry.id,
                    "title": entry.title,
                    "category": entry.category,
                    "sub_category": entry.sub_category,
                    "resolution_script": entry.resolution_script,
                    "success_rate": entry.success_rate
                })
            
            return results
        finally:
            db.close()
    
    async def get_entry(self, entry_id: int) -> Optional[KnowledgeBaseEntry]:
        """Get knowledge base entry by ID"""
        db = self._get_db()
        try:
            return db.query(KnowledgeBaseEntry).filter(
                KnowledgeBaseEntry.id == entry_id
            ).first()
        finally:
            db.close()
    
    async def create_entry(self, entry_data: Dict[str, Any]) -> KnowledgeBaseEntry:
        """Create knowledge base entry"""
        db = self._get_db()
        try:
            entry = KnowledgeBaseEntry(
                title=entry_data.get("title"),
                content=entry_data.get("content"),
                category=entry_data.get("category"),
                sub_category=entry_data.get("sub_category"),
                resolution_steps=entry_data.get("resolution_steps"),
                resolution_script=entry_data.get("resolution_script"),
                tags=entry_data.get("tags")
            )
            
            db.add(entry)
            db.commit()
            db.refresh(entry)
            
            return entry
        finally:
            db.close()

