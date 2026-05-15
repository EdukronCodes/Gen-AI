from sqlalchemy.orm import Session
from app.database.models import Refund


class RefundRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_refund_number(self, refund_number: str) -> Refund | None:
        return self.db.query(Refund).filter(Refund.refund_number == refund_number.upper()).first()

    def get_by_order(self, order_id: int):
        return self.db.query(Refund).filter(Refund.order_id == order_id).all()

    def create(self, refund: Refund) -> Refund:
        self.db.add(refund)
        self.db.commit()
        self.db.refresh(refund)
        return refund
