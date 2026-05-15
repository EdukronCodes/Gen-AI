from sqlalchemy.orm import Session
from app.services.order_service import OrderService
from app.agents.orchestrator.agent_router import extract_order_number


class OrderTrackingAgent:
    def __init__(self, db: Session):
        self.order_service = OrderService(db)

    def handle(self, context: dict) -> dict:
        message = context["message"]
        order_number = context.get("order_number") or extract_order_number(message)

        if not order_number:
            return {
                "reply": (
                    "I'd be happy to track your order! Please provide your order number "
                    "(e.g. ORD-10001). You can find it in your confirmation email."
                ),
                "metadata": {},
            }

        order = self.order_service.track_order(order_number)
        if not order:
            return {
                "reply": f"I couldn't find order **{order_number}**. Please double-check the number and try again.",
                "metadata": {"order_number": order_number},
            }

        status_emoji = {"delivered": "✅", "shipped": "🚚", "processing": "⏳", "cancelled": "❌"}.get(order["status"], "📦")
        items_str = ", ".join(f"{i['product']} x{i['qty']}" for i in order["items"])
        reply = (
            f"{status_emoji} **Order {order['order_number']}**\n"
            f"• Status: **{order['status'].title()}**\n"
            f"• Total: ${order['total_amount']:.2f}\n"
            f"• Items: {items_str}\n"
        )
        if order["tracking_number"]:
            reply += f"• Tracking: {order['tracking_number']} ({order['carrier'] or 'Carrier TBD'})\n"
        if order["estimated_delivery"]:
            reply += f"• Est. delivery: {order['estimated_delivery'][:10]}\n"
        return {"reply": reply.strip(), "metadata": order}
