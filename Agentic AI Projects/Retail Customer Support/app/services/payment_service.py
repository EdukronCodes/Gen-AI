import logging

logger = logging.getLogger(__name__)


class PaymentService:
    def process_refund(self, amount: float, order_number: str) -> dict:
        logger.info("Processing refund $%.2f for order %s", amount, order_number)
        return {"success": True, "transaction_id": f"TXN-{order_number}-REF", "amount": amount}
