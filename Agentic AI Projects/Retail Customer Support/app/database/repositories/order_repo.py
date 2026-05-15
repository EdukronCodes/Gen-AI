from sqlalchemy.orm import Session, joinedload
from app.database.models import Order, OrderItem


class OrderRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_order_number(self, order_number: str) -> Order | None:
        return (
            self.db.query(Order)
            .options(joinedload(Order.items).joinedload(OrderItem.product))
            .filter(Order.order_number == order_number.upper())
            .first()
        )

    def get_by_customer(self, customer_id: int):
        return self.db.query(Order).filter(Order.customer_id == customer_id).all()

    def update_status(self, order: Order, status: str) -> Order:
        order.status = status
        self.db.commit()
        self.db.refresh(order)
        return order
