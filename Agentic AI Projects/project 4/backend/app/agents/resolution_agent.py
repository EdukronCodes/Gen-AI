"""
Resolution Agent
Executes auto-fix scripts and attempts self-healing
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.services.ticket_service import TicketService
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.script_executor import ScriptExecutor
from datetime import datetime


class ResolutionAgent(BaseAgent):
    """Attempts auto-resolution using scripts and knowledge base"""
    
    def __init__(self):
        super().__init__(
            name="Resolution Agent",
            role="Auto-Resolver",
            goal="Automatically resolve tickets using scripts and knowledge base"
        )
        self.ticket_service = TicketService()
        self.kb_service = KnowledgeBaseService()
        self.script_executor = ScriptExecutor()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt auto-resolution"""
        ticket_id = input_data.get("ticket_id")
        ticket = await self.ticket_service.get_ticket(ticket_id)
        
        if not ticket:
            return {"error": "Ticket not found", **input_data}
        
        # Search knowledge base for resolution scripts
        resolution_candidates = await self.kb_service.find_resolution_scripts(
            category=ticket.category,
            sub_category=ticket.sub_category,
            root_symptom=ticket.root_symptom
        )
        
        if not resolution_candidates:
            return {
                **input_data,
                "auto_resolved": "false",
                "reason": "No auto-fix script available"
            }
        
        # Select best resolution script using LLM
        resolution_prompt = f"""
        Select the best auto-fix script for this ticket:
        
        Ticket:
        - Category: {ticket.category}
        - Sub-category: {ticket.sub_category}
        - Root Symptom: {ticket.root_symptom}
        - Description: {ticket.description}
        
        Available Scripts:
        {self._format_resolution_scripts(resolution_candidates)}
        
        Return JSON with script_id and confidence (0-100).
        """
        
        response = await self.llm.ainvoke(resolution_prompt)
        selection = self._parse_llm_response(response.content)
        
        script_id = selection.get("script_id")
        confidence = selection.get("confidence", 0)
        
        # Only auto-resolve if confidence is high (>70%)
        if confidence < 70 or not script_id:
            return {
                **input_data,
                "auto_resolved": "false",
                "reason": f"Low confidence ({confidence}%)"
            }
        
        # Get script details
        script = next((s for s in resolution_candidates if s.get("id") == script_id), None)
        if not script:
            return {
                **input_data,
                "auto_resolved": "false",
                "reason": "Script not found"
            }
        
        # Execute script
        try:
            execution_result = await self.script_executor.execute(
                script.get("resolution_script"),
                ticket_id=ticket_id
            )
            
            if execution_result.get("success"):
                # Auto-resolve ticket
                await self.ticket_service.resolve_ticket(
                    ticket_id,
                    resolution=execution_result.get("output", "Auto-resolved by Resolution Agent"),
                    auto_resolved=True,
                    resolution_script=script.get("resolution_script")
                )
                
                self.log_action("Ticket auto-resolved", {
                    "ticket_id": ticket_id,
                    "script_id": script_id,
                    "confidence": confidence
                })
                
                return {
                    **input_data,
                    "auto_resolved": "true",
                    "resolution_script": script.get("resolution_script"),
                    "execution_result": execution_result
                }
            else:
                return {
                    **input_data,
                    "auto_resolved": "false",
                    "reason": f"Script execution failed: {execution_result.get('error')}"
                }
        
        except Exception as e:
            return {
                **input_data,
                "auto_resolved": "false",
                "reason": f"Execution error: {str(e)}"
            }
    
    def _format_resolution_scripts(self, scripts: list) -> str:
        """Format scripts for LLM context"""
        formatted = []
        for script in scripts:
            formatted.append(
                f"Script ID {script.get('id')}:\n"
                f"  - Title: {script.get('title')}\n"
                f"  - Success Rate: {script.get('success_rate', 0)}%\n"
                f"  - Script: {script.get('resolution_script', '')[:200]}"
            )
        return "\n\n".join(formatted)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM script selection response"""
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
                "script_id": data.get("script_id"),
                "confidence": int(data.get("confidence", 0))
            }
        except:
            return {"script_id": None, "confidence": 0}


