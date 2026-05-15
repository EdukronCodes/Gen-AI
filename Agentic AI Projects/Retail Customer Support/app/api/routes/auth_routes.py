from fastapi import APIRouter
from pydantic import BaseModel
from datetime import timedelta
from app.security.jwt_handler import create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    email: str
    password: str = "demo"


@router.post("/login")
def login(request: LoginRequest):
    # Demo auth - accepts any email with password "demo"
    if request.password != "demo":
        return {"error": "Invalid credentials. Use password: demo"}
    token = create_access_token({"sub": request.email, "role": "admin"})
    return {"access_token": token, "token_type": "bearer"}
