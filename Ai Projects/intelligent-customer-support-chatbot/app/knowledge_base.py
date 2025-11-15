"""
Knowledge Base management with vector database for semantic search
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from config import settings
import uuid


class KnowledgeBase:
    """Knowledge Base with vector search capabilities"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample FAQ data"""
        if self.collection.count() == 0:
            sample_faqs = [
                {
                    "title": "How do I track my order?",
                    "content": "You can track your order by logging into your account and going to the Orders section. You'll receive a tracking number via email once your order ships.",
                    "category": "order_status"
                },
                {
                    "title": "What is your return policy?",
                    "content": "We offer a 30-day return policy. Items must be in original condition with tags attached. Returns are free for orders over $50.",
                    "category": "return_refund"
                },
                {
                    "title": "How do I reset my password?",
                    "content": "Click on 'Forgot Password' on the login page. Enter your email address and you'll receive a password reset link within 5 minutes.",
                    "category": "account_management"
                },
                {
                    "title": "What payment methods do you accept?",
                    "content": "We accept all major credit cards (Visa, Mastercard, American Express), PayPal, and Apple Pay. All payments are processed securely.",
                    "category": "billing_payment"
                },
                {
                    "title": "How long does shipping take?",
                    "content": "Standard shipping takes 5-7 business days. Express shipping (2-3 business days) and overnight shipping are also available at checkout.",
                    "category": "general_faq"
                },
                {
                    "title": "Can I cancel my order?",
                    "content": "You can cancel your order within 24 hours of placing it. After that, you'll need to wait for delivery and use our return process.",
                    "category": "order_status"
                },
                {
                    "title": "Do you ship internationally?",
                    "content": "Yes, we ship to over 50 countries. International shipping rates and delivery times vary by location. Check our shipping page for details.",
                    "category": "general_faq"
                },
                {
                    "title": "How do I contact customer service?",
                    "content": "You can reach us via this chatbot 24/7, email at support@company.com, or phone at 1-800-XXX-XXXX during business hours (9 AM - 6 PM EST).",
                    "category": "general_faq"
                }
            ]
            
            for faq in sample_faqs:
                self.add_article(
                    title=faq["title"],
                    content=faq["content"],
                    category=faq["category"]
                )
    
    def add_article(self, title: str, content: str, category: str, tags: Optional[List[str]] = None):
        """Add article to knowledge base"""
        article_id = str(uuid.uuid4())
        full_text = f"{title} {content}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(full_text).tolist()
        
        # Store in ChromaDB
        self.collection.add(
            ids=[article_id],
            embeddings=[embedding],
            documents=[full_text],
            metadatas=[{
                "title": title,
                "content": content,
                "category": category,
                "tags": ",".join(tags or [])
            }]
        )
        
        return article_id
    
    def search(self, query: str, category: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Search knowledge base using semantic similarity"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build where clause for category filter
        where_clause = {}
        if category:
            where_clause["category"] = category
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "article_id": results["ids"][0][i],
                    "title": results["metadatas"][0][i].get("title", ""),
                    "content": results["metadatas"][0][i].get("content", ""),
                    "category": results["metadatas"][0][i].get("category", ""),
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def get_article(self, article_id: str) -> Optional[Dict]:
        """Get article by ID"""
        results = self.collection.get(ids=[article_id])
        if results["ids"]:
            return {
                "article_id": results["ids"][0],
                "title": results["metadatas"][0].get("title", ""),
                "content": results["metadatas"][0].get("content", ""),
                "category": results["metadatas"][0].get("category", "")
            }
        return None
    
    def delete_article(self, article_id: str) -> bool:
        """Delete article from knowledge base"""
        try:
            self.collection.delete(ids=[article_id])
            return True
        except Exception as e:
            print(f"Error deleting article: {e}")
            return False

