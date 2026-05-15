from sqlalchemy.orm import Session
from app.rag.retrieval_pipeline import RetrievalPipeline


class CustomerSupportAgent:
    def __init__(self, db: Session):
        self.retrieval = RetrievalPipeline()

    def handle(self, context: dict) -> dict:
        message = context["message"]
        docs = self.retrieval.retrieve(message, top_k=2)
        context_text = "\n".join(d["content"] for d in docs) if docs else ""

        reply = (
            "Hello! I'm your retail support assistant. I can help you with:\n"
            "• **Order tracking** — e.g. \"Where is order ORD-10002?\"\n"
            "• **Refunds & returns** — e.g. \"I want a refund for my headphones\"\n"
            "• **Product recommendations** — e.g. \"Show me electronics\"\n"
            "• **General questions** about our store policies\n\n"
        )
        if context_text:
            reply += f"Based on our knowledge base:\n{context_text}\n\n"
        reply += "How can I assist you today?"
        return {"reply": reply, "metadata": {"docs_used": len(docs)}}
