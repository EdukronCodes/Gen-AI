from sqlalchemy.orm import Session
from app.database.models import Product


class InventoryService:
    def __init__(self, db: Session):
        self.db = db

    def check_stock(self, sku: str) -> dict:
        product = self.db.query(Product).filter(Product.sku == sku.upper()).first()
        if not product:
            return {"available": False, "message": "Product not found"}
        return {
            "sku": product.sku,
            "name": product.name,
            "in_stock": product.stock_quantity > 0,
            "quantity": product.stock_quantity,
        }
