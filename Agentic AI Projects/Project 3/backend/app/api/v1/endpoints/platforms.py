"""
Platform connection API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from app.core.database import get_db
from app.models.platform import PlatformConnection

router = APIRouter()


class PlatformConnectionCreate(BaseModel):
    user_id: int
    platform: str
    access_token: str
    refresh_token: str = None
    platform_user_id: str = None
    platform_username: str = None


class PlatformConnectionResponse(BaseModel):
    id: int
    platform: str
    platform_username: str
    is_active: bool
    
    class Config:
        from_attributes = True


@router.post("/connect", response_model=PlatformConnectionResponse)
def connect_platform(
    connection: PlatformConnectionCreate,
    db: Session = Depends(get_db)
):
    """Connect a social media platform"""
    db_connection = PlatformConnection(**connection.dict())
    db.add(db_connection)
    db.commit()
    db.refresh(db_connection)
    return db_connection


@router.get("/", response_model=List[PlatformConnectionResponse])
def get_connections(
    user_id: int = None,
    db: Session = Depends(get_db)
):
    """Get platform connections"""
    query = db.query(PlatformConnection).filter(PlatformConnection.is_active == True)
    
    if user_id:
        query = query.filter(PlatformConnection.user_id == user_id)
    
    return query.all()


@router.delete("/{connection_id}")
def disconnect_platform(connection_id: int, db: Session = Depends(get_db)):
    """Disconnect a platform"""
    connection = db.query(PlatformConnection).filter(
        PlatformConnection.id == connection_id
    ).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    connection.is_active = False
    db.commit()
    
    return {"message": "Platform disconnected"}

