import re
from sqlalchemy.orm import Session
from app.services.recommendation_service import RecommendationService


class RecommendationAgent:
    CATEGORIES = ["electronics", "apparel", "footwear", "home"]

    def __init__(self, db: Session):
        self.rec_service = RecommendationService(db)

    def handle(self, context: dict) -> dict:
        message = context["message"].lower()
        category = next((c for c in self.CATEGORIES if c in message), None)

        if category:
            products = self.rec_service.recommend_by_category(category)
        else:
            query = re.sub(r"\b(recommend|suggest|show|find|looking for|me)\b", "", message).strip()
            products = self.rec_service.search_products(query) if len(query) > 2 else self.rec_service.get_popular()

        if not products:
            products = self.rec_service.get_popular()

        lines = ["Here are some products you might like:\n"]
        for p in products:
            stock = "In stock" if p["in_stock"] else "Out of stock"
            lines.append(f"• **{p['name']}** — ${p['price']:.2f} ({p['category']}) — {stock}")
        return {"reply": "\n".join(lines), "metadata": {"products": products}}
