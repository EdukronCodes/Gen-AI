from sqlalchemy.orm import Session
from app.database.repositories.order_repo import OrderRepository


class OrderService:
    def __init__(self, db: Session):
        self.repo = OrderRepository(db)

    def track_order(self, order_number: str) -> dict | None:
        order = self.repo.get_by_order_number(order_number)
        if not order:
            return None
        items = [
            {"product": i.product.name if i.product else "Unknown", "qty": i.quantity, "price": i.unit_price}
            for i in order.items
        ]
        return {
            "order_number": order.order_number,
            "status": order.status,
            "total_amount": order.total_amount,
            "tracking_number": order.tracking_number,
            "carrier": order.carrier,
            "estimated_delivery": order.estimated_delivery.isoformat() if order.estimated_delivery else None,
            "items": items,
        }

    def cancel_order(self, order_number: str) -> dict:
        order = self.repo.get_by_order_number(order_number)
        if not order:
            return {"success": False, "message": "Order not found"}
        if order.status in ("delivered", "cancelled"):
            return {"success": False, "message": f"Cannot cancel order in '{order.status}' status"}
        self.repo.update_status(order, "cancelled")
        return {"success": True, "message": f"Order {order_number} has been cancelled", "status": "cancelled"}
