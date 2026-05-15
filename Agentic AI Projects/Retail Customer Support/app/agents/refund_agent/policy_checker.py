class PolicyChecker:
    REFUND_WINDOW_DAYS = 30
    NON_REFUNDABLE_STATUSES = {"cancelled"}

    def check_eligibility(self, order_status: str, days_since_order: int = 0) -> tuple[bool, str]:
        if order_status in self.NON_REFUNDABLE_STATUSES:
            return False, "Cancelled orders are handled separately."
        if order_status == "delivered" and days_since_order > self.REFUND_WINDOW_DAYS:
            return False, f"Return window of {self.REFUND_WINDOW_DAYS} days has expired."
        if order_status in ("delivered", "shipped", "processing"):
            return True, "Eligible under our 30-day return policy."
        return False, f"Orders with status '{order_status}' cannot be refunded yet."
