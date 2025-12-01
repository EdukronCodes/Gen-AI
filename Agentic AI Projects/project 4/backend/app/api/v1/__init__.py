from fastapi import APIRouter
from app.api.v1 import tickets, orchestrator, health, auth

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["authentication"])
router.include_router(tickets.router, prefix="/tickets", tags=["tickets"])
router.include_router(orchestrator.router, prefix="/orchestrator", tags=["orchestrator"])
router.include_router(health.router, prefix="/health", tags=["health"])

