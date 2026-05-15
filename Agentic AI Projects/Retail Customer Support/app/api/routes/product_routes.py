from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.config.database import get_db
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/api/v1/products", tags=["Products"])


@router.get("/search")
def search_products(q: str = Query(..., min_length=1), limit: int = 10, db: Session = Depends(get_db)):
    service = RecommendationService(db)
    return {"products": service.search_products(q, limit)}


@router.get("/recommend")
def recommend(category: str | None = None, db: Session = Depends(get_db)):
    service = RecommendationService(db)
    if category:
        return {"products": service.recommend_by_category(category)}
    return {"products": service.get_popular()}
