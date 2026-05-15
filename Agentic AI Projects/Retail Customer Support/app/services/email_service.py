import logging

logger = logging.getLogger(__name__)


class EmailService:
    def send(self, to: str, subject: str, body: str) -> bool:
        logger.info("Email sent to %s | Subject: %s", to, subject)
        return True
