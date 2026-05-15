import logging

logger = logging.getLogger(__name__)


class NotificationService:
    """Orchestrates notifications across channels."""

    def send_order_update(self, email: str, order_number: str, message: str) -> bool:
        logger.info("Order notification to %s for %s: %s", email, order_number, message)
        return True

    def send_refund_update(self, email: str, refund_number: str, message: str) -> bool:
        logger.info("Refund notification to %s for %s: %s", email, refund_number, message)
        return True
