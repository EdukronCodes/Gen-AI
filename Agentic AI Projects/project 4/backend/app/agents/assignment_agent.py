"""
Assignment Agent
Auto-assigns engineers based on skills, workload, and performance
"""
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.services.ticket_service import TicketService
from app.services.engineer_service import EngineerService


class AssignmentAgent(BaseAgent):
    """Auto-assigns tickets to best available engineer"""
    
    def __init__(self):
        super().__init__(
            name="Assignment Agent",
            role="Resource Allocator",
            goal="Assign tickets to the most suitable engineer automatically"
        )
        self.ticket_service = TicketService()
        self.engineer_service = EngineerService()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assign ticket to engineer"""
        ticket_id = input_data.get("ticket_id")
        ticket = await self.ticket_service.get_ticket(ticket_id)
        
        if not ticket:
            return {"error": "Ticket not found", **input_data}
        
        # Get available engineers
        engineers = await self.engineer_service.get_available_engineers()
        
        if not engineers:
            return {
                **input_data,
                "assigned": False,
                "error": "No available engineers"
            }
        
        # Score engineers using LLM
        assignment_prompt = f"""
        Assign this ticket to the best engineer:
        
        Ticket:
        - Category: {ticket.category}
        - Sub-category: {ticket.sub_category}
        - Priority: {ticket.priority}
        - Description: {ticket.description}
        
        Available Engineers:
        {self._format_engineers(engineers)}
        
        Consider:
        1. Skill match with ticket category
        2. Current workload (lower is better)
        3. Past performance (avg resolution time)
        4. Auto-resolution rate
        
        Return JSON with engineer_id and reasoning.
        """
        
        response = await self.llm.ainvoke(assignment_prompt)
        assignment_data = self._parse_llm_response(response.content)
        
        selected_engineer_id = assignment_data.get("engineer_id")
        
        # Validate engineer exists and is available
        selected_engineer = next(
            (e for e in engineers if e.id == selected_engineer_id),
            None
        )
        
        if not selected_engineer or selected_engineer.active_tickets_count >= selected_engineer.max_concurrent_tickets:
            # Fallback: assign to engineer with lowest workload
            selected_engineer = min(engineers, key=lambda e: e.active_tickets_count)
            selected_engineer_id = selected_engineer.id
        
        # Assign ticket
        await self.ticket_service.assign_ticket(ticket_id, selected_engineer_id)
        await self.engineer_service.increment_workload(selected_engineer_id)
        
        self.log_action("Ticket assigned", {
            "ticket_id": ticket_id,
            "engineer_id": selected_engineer_id,
            "engineer_name": selected_engineer.full_name
        })
        
        return {
            **input_data,
            "assigned_engineer_id": selected_engineer_id,
            "assigned_engineer_name": selected_engineer.full_name,
            "status": "assigned"
        }
    
    def _format_engineers(self, engineers: List) -> str:
        """Format engineers for LLM context"""
        formatted = []
        for eng in engineers:
            skills = ", ".join([s.skill_name for s in eng.skills[:3]])
            formatted.append(
                f"Engineer ID {eng.id}: {eng.full_name}\n"
                f"  - Skills: {skills}\n"
                f"  - Active Tickets: {eng.active_tickets_count}/{eng.max_concurrent_tickets}\n"
                f"  - Avg Resolution: {eng.avg_resolution_time_hours}h\n"
                f"  - Auto-Resolution Rate: {eng.auto_resolution_rate}%"
            )
        return "\n\n".join(formatted)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM assignment response"""
        import json
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
            else:
                json_str = response
            
            data = json.loads(json_str)
            engineer_id = int(data.get("engineer_id", 0))
            return {
                "engineer_id": engineer_id,
                "reasoning": data.get("reasoning", "")
            }
        except:
            return {"engineer_id": None, "reasoning": "Error parsing response"}


