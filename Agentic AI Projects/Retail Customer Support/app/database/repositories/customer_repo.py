from sqlalchemy.orm import Session
from app.database.models import Customer


class CustomerRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str) -> Customer | None:
        return self.db.query(Customer).filter(Customer.email == email).first()

    def get_by_id(self, customer_id: int) -> Customer | None:
        return self.db.query(Customer).filter(Customer.id == customer_id).first()

    def list_all(self, limit: int = 50):
        return self.db.query(Customer).limit(limit).all()
