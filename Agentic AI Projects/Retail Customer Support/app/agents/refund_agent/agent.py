from sqlalchemy.orm import Session
from app.database.repositories.order_repo import OrderRepository
from app.database.repositories.refund_repo import RefundRepository
from app.agents.orchestrator.agent_router import extract_order_number, extract_refund_number
from app.agents.refund_agent.policy_checker import PolicyChecker


class RefundAgent:
    def __init__(self, db: Session):
        self.order_repo = OrderRepository(db)
        self.refund_repo = RefundRepository(db)
        self.policy = PolicyChecker()

    def handle(self, context: dict) -> dict:
        message = context["message"]
        refund_num = extract_refund_number(message)
        order_num = context.get("order_number") or extract_order_number(message)

        if refund_num:
            refund = self.refund_repo.get_by_refund_number(refund_num)
            if refund:
                return {
                    "reply": (
                        f"**Refund {refund.refund_number}**\n"
                        f"• Status: **{refund.status.title()}**\n"
                        f"• Amount: ${refund.amount:.2f}\n"
                        f"• Reason: {refund.reason}"
                    ),
                    "metadata": {"refund_number": refund_num},
                }
            return {"reply": f"Refund {refund_num} not found.", "metadata": {}}

        if order_num:
            order = self.order_repo.get_by_order_number(order_num)
            if not order:
                return {"reply": f"Order {order_num} not found.", "metadata": {}}
            eligible, reason = self.policy.check_eligibility(order.status, days_since_order=10)
            if eligible:
                return {
                    "reply": (
                        f"Your order **{order_num}** is eligible for a refund.\n"
                        f"• Order total: ${order.total_amount:.2f}\n"
                        f"• Policy: {reason}\n\n"
                        "I've initiated a refund request. You'll receive confirmation within 3-5 business days."
                    ),
                    "metadata": {"order_number": order_num, "eligible": True},
                }
            return {
                "reply": f"Unfortunately, order **{order_num}** is not eligible: {reason}",
                "metadata": {"eligible": False},
            }

        return {
            "reply": (
                "I can help with refunds and returns. Please share your order number (e.g. ORD-10001) "
                "or refund reference (e.g. REF-5001).\n\n"
                "**Our policy:** 30-day returns on unused items. Damaged items refunded immediately."
            ),
            "metadata": {},
        }
