FAQ_KNOWLEDGE = [
    {"id": "faq-1", "content": "Free shipping on orders over $50. Standard delivery 3-5 business days."},
    {"id": "faq-2", "content": "30-day return policy for unused items in original packaging. Refunds processed in 3-5 business days."},
    {"id": "faq-3", "content": "Order tracking available once shipped. Use order number (ORD-XXXXX) in chat or account page."},
    {"id": "faq-4", "content": "Customer support hours: Mon-Fri 9am-6pm EST, Sat 10am-4pm EST."},
    {"id": "faq-5", "content": "Damaged items: contact us within 48 hours with photos for immediate replacement or refund."},
    {"id": "faq-6", "content": "Payment methods: Visa, Mastercard, Amex, PayPal, Apple Pay."},
]


class RetrievalPipeline:
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_words = set(query.lower().split())
        scored = []
        for doc in FAQ_KNOWLEDGE:
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
