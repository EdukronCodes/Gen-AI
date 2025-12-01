"""
Classification Agent
Understands user intent and classifies issues using LLM + RAG
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.services.ticket_service import TicketService
from app.services.rag_service import RAGService
from app.services.knowledge_base_service import KnowledgeBaseService


class ClassificationAgent(BaseAgent):
    """Classifies tickets using AI and knowledge base"""
    
    def __init__(self):
        super().__init__(
            name="Classification Agent",
            role="Issue Classifier",
            goal="Accurately classify IT issues into categories and predict resolution"
        )
        self.ticket_service = TicketService()
        self.rag_service = RAGService()
        self.kb_service = KnowledgeBaseService()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify ticket using LLM + RAG"""
        ticket_id = input_data.get("ticket_id")
        ticket = await self.ticket_service.get_ticket(ticket_id)
        
        if not ticket:
            return {"error": "Ticket not found", **input_data}
        
        # Search knowledge base using RAG
        similar_cases = await self.rag_service.search_similar_cases(
            ticket.description,
            top_k=5
        )
        
        # Classify using LLM with context
        classification_prompt = f"""
        Classify this IT help desk ticket:
        
        Title: {ticket.title}
        Description: {ticket.description}
        
        Similar cases from knowledge base:
        {self._format_similar_cases(similar_cases)}
        
        Classify into:
        1. Category: network, server, application, database, cloud, security, other
        2. Sub-category: specific area
        3. Root symptom: what's the actual problem
        4. Severity: critical, high, medium, low
        5. Probable resolution: likely fix approach
        
        Return JSON format.
        """
        
        response = await self.llm.ainvoke(classification_prompt)
        classification = self._parse_llm_response(response.content)
        
        # Update ticket
        update_data = {
            "category": classification.get("category"),
            "sub_category": classification.get("sub_category"),
            "root_symptom": classification.get("root_symptom"),
            "status": "classified",
            "agent_metadata": {
                "classification": classification,
                "similar_cases_found": len(similar_cases)
            }
        }
        
        await self.ticket_service.update_ticket(ticket_id, update_data)
        
        self.log_action("Ticket classified", {
            "ticket_id": ticket_id,
            "category": classification.get("category")
        })
        
        return {
            **input_data,
            "classification": classification,
            "status": "classified"
        }
    
    def _format_similar_cases(self, cases: list) -> str:
        """Format similar cases for LLM context"""
        if not cases:
            return "No similar cases found."
        
        formatted = []
        for case in cases[:3]:  # Top 3
            formatted.append(f"- {case.get('title', '')}: {case.get('content', '')[:200]}")
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM classification response"""
        import json
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
            else:
                json_str = response
            
            data = json.loads(json_str)
            return {
                "category": data.get("category", "other"),
                "sub_category": data.get("sub_category", ""),
                "root_symptom": data.get("root_symptom", ""),
                "severity": data.get("severity", "medium"),
                "probable_resolution": data.get("probable_resolution", "")
            }
        except:
            return {
                "category": "other",
                "sub_category": "",
                "root_symptom": "",
                "severity": "medium",
                "probable_resolution": ""
            }


