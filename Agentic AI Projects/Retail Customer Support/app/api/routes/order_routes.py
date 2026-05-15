from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.config.database import get_db
from app.services.order_service import OrderService

router = APIRouter(prefix="/api/v1/orders", tags=["Orders"])


@router.get("/{order_number}")
def track_order(order_number: str, db: Session = Depends(get_db)):
    service = OrderService(db)
    order = service.track_order(order_number.upper())
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@router.post("/{order_number}/cancel")
def cancel_order(order_number: str, db: Session = Depends(get_db)):
    service = OrderService(db)
    result = service.cancel_order(order_number.upper())
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result
