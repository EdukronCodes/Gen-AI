"""
Post API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from pydantic import BaseModel

from app.core.database import get_db
from app.agents.posting_agent import PostingAgent
from app.models.platform import Post

router = APIRouter()
posting_agent = PostingAgent()


class PostCreate(BaseModel):
    platform: str
    content: Dict[str, Any]
    media_paths: List[str] = []
    scheduled_time: str = None


class PostResponse(BaseModel):
    id: int
    platform: str
    platform_post_id: str = None
    status: str
    
    class Config:
        from_attributes = True


@router.post("/", response_model=Dict[str, Any])
async def create_post(
    post: PostCreate,
    platform_connection: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Create and post content"""
    try:
        context = {
            "platform": post.platform,
            "content": post.content,
            "media_paths": post.media_paths,
            "scheduled_time": post.scheduled_time,
            "platform_connection": platform_connection
        }
        
        result = await posting_agent.execute(context)
        
        # Save to database
        db_post = Post(
            user_id=platform_connection.get("user_id", 1),
            platform_connection_id=platform_connection.get("id"),
            platform=post.platform,
            platform_post_id=result.get("platform_post_id"),
            caption=post.content.get("caption", ""),
            hashtags=post.content.get("hashtags", [])
        )
        
        db.add(db_post)
        db.commit()
        db.refresh(db_post)
        
        return {
            **result,
            "db_post_id": db_post.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[PostResponse])
def get_posts(
    platform: str = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get posts"""
    query = db.query(Post)
    
    if platform:
        query = query.filter(Post.platform == platform)
    
    posts = query.offset(skip).limit(limit).all()
    return posts


@router.get("/{post_id}", response_model=PostResponse)
def get_post(post_id: int, db: Session = Depends(get_db)):
    """Get post by ID"""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

