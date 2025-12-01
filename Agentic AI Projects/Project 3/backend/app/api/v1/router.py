"""
Main API router
"""
from fastapi import APIRouter
from app.api.v1.endpoints import campaigns, platforms, analytics, posts

api_router = APIRouter()

api_router.include_router(campaigns.router, prefix="/campaigns", tags=["Campaigns"])
api_router.include_router(platforms.router, prefix="/platforms", tags=["Platforms"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(posts.router, prefix="/posts", tags=["Posts"])

