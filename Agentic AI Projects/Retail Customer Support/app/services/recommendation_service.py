from sqlalchemy.orm import Session
from app.database.models import Product


class RecommendationService:
    def __init__(self, db: Session):
        self.db = db

    def search_products(self, query: str, limit: int = 5) -> list[dict]:
        q = f"%{query.lower()}%"
        products = (
            self.db.query(Product)
            .filter(
                (Product.name.ilike(q))
                | (Product.description.ilike(q))
                | (Product.category.ilike(q))
            )
            .limit(limit)
            .all()
        )
        return [self._to_dict(p) for p in products]

    def recommend_by_category(self, category: str, limit: int = 4) -> list[dict]:
        products = (
            self.db.query(Product)
            .filter(Product.category.ilike(f"%{category}%"))
            .order_by(Product.stock_quantity.desc())
            .limit(limit)
            .all()
        )
        return [self._to_dict(p) for p in products]

    def get_popular(self, limit: int = 4) -> list[dict]:
        products = self.db.query(Product).order_by(Product.stock_quantity.desc()).limit(limit).all()
        return [self._to_dict(p) for p in products]

    @staticmethod
    def _to_dict(p: Product) -> dict:
        return {
            "id": p.id,
            "sku": p.sku,
            "name": p.name,
            "description": p.description,
            "category": p.category,
            "price": p.price,
            "in_stock": p.stock_quantity > 0,
        }
