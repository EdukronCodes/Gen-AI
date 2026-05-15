from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.config.database import get_db
from app.database.repositories.refund_repo import RefundRepository

router = APIRouter(prefix="/api/v1/refunds", tags=["Refunds"])


@router.get("/{refund_number}")
def get_refund_status(refund_number: str, db: Session = Depends(get_db)):
    repo = RefundRepository(db)
    refund = repo.get_by_refund_number(refund_number.upper())
    if not refund:
        raise HTTPException(status_code=404, detail="Refund not found")
    return {
        "refund_number": refund.refund_number,
        "order_id": refund.order_id,
        "amount": refund.amount,
        "reason": refund.reason,
        "status": refund.status,
        "created_at": refund.created_at.isoformat(),
    }
