"""
RAG Service
Retrieval Augmented Generation for knowledge base search
"""
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
import os


class RAGService:
    """RAG service for knowledge base search"""
    
    def __init__(self):
        # Initialize embeddings
        if settings.USE_AZURE:
            from langchain_openai import AzureOpenAIEmbeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                azure_deployment="text-embedding-ada-002"
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key=settings.OPENAI_API_KEY
            )
        
        # Initialize vector store
        persist_directory = settings.CHROMA_PERSIST_DIR
        os.makedirs(persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    async def search_similar_cases(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar cases in knowledge base"""
        try:
            # Search vector store
            docs = self.vectorstore.similarity_search(query, k=top_k)
            
            results = []
            for doc in docs:
                results.append({
                    "title": doc.metadata.get("title", ""),
                    "content": doc.page_content,
                    "category": doc.metadata.get("category", ""),
                    "score": doc.metadata.get("score", 0)
                })
            
            return results
        except Exception as e:
            print(f"RAG search error: {e}")
            return []
    
    async def add_to_knowledge_base(self, title: str, content: str, 
                                   category: str, metadata: Dict[str, Any] = None):
        """Add document to knowledge base"""
        try:
            # Split text
            texts = self.text_splitter.split_text(content)
            
            # Add metadata
            metadatas = []
            for text in texts:
                meta = {
                    "title": title,
                    "category": category,
                    **(metadata or {})
                }
                metadatas.append(meta)
            
            # Add to vector store
            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self.vectorstore.persist()
            
            return True
        except Exception as e:
            print(f"Error adding to knowledge base: {e}")
            return False


