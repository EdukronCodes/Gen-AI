"""
Intake Agent
Accepts tickets from multiple channels and creates structured incidents
"""
from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.services.ticket_service import TicketService
from app.services.kafka_service import KafkaService
from datetime import datetime
import uuid


class IntakeAgent(BaseAgent):
    """Handles ticket intake from all channels"""
    
    def __init__(self):
        super().__init__(
            name="Intake Agent",
            role="Ticket Intake Specialist",
            goal="Extract structured information from user inputs across all channels"
        )
        self.ticket_service = TicketService()
        self.kafka_service = KafkaService()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process intake from any channel"""
        channel = input_data.get("channel", "web")  # web, chat, email, voice, monitoring
        
        # Extract information using LLM
        prompt = f"""
        Extract structured information from this IT help desk request:
        
        Input: {input_data.get('raw_input', '')}
        Channel: {channel}
        
        Extract:
        1. Problem description
        2. Device/System affected
        3. Location
        4. Impact level (high/medium/low)
        5. Urgency indicator
        
        Return JSON format.
        """
        
        response = await self.llm.ainvoke(prompt)
        extracted_data = self._parse_llm_response(response.content)
        
        # Create ticket
        ticket_data = {
            "title": extracted_data.get("title", input_data.get("title", "IT Issue")),
            "description": extracted_data.get("problem_description", input_data.get("raw_input", "")),
            "source_channel": channel,
            "device_info": extracted_data.get("device_info"),
            "location": extracted_data.get("location"),
            "impact": extracted_data.get("impact", "medium"),
            "status": "created"
        }
        
        ticket = await self.ticket_service.create_ticket(ticket_data)
        
        # Publish to Kafka
        await self.kafka_service.publish_ticket_event({
            "event_type": "ticket_created",
            "ticket_id": ticket.id,
            "ticket_number": ticket.ticket_number,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self.log_action("Ticket created", {"ticket_id": ticket.id, "ticket_number": ticket.ticket_number})
        
        return {
            "ticket_id": ticket.id,
            "ticket_number": ticket.ticket_number,
            "extracted_data": extracted_data,
            "status": "created"
        }
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        import json
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
            else:
                json_str = response
            
            return json.loads(json_str)
        except:
            # Fallback parsing
            return {
                "problem_description": response,
                "title": "IT Issue",
                "device_info": None,
                "location": None,
                "impact": "medium"
            }


