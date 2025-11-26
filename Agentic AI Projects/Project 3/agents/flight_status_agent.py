"""
Flight Status Agent - Provides real-time flight status information
"""
from typing import Dict, Any
from sqlalchemy.orm import Session
from .base_agent import BaseAgent
from database.models import Flight, FlightStatus, FlightStatusUpdate, Airport


class FlightStatusAgent(BaseAgent):
    """Agent specialized in flight status information"""
    
    def __init__(self, db: Session):
        super().__init__(
            name="Flight Status Agent",
            description="an expert at providing real-time flight status, delays, gate information, and arrival times",
            db=db
        )
    
    def get_knowledge_documents(self) -> list[str]:
        return [
            "Flight statuses include: scheduled, on_time, delayed, boarding, departed, arrived, cancelled.",
            "Delays can occur due to weather, air traffic, mechanical issues, or crew availability.",
            "Gate information is typically available 2-3 hours before departure.",
            "Flight status updates are provided in real-time.",
            "Passengers should check flight status before heading to the airport.",
        ]
    
    def get_flight_status(self, flight_number: str = None, flight_id: int = None) -> Dict[str, Any]:
        """Get current status of a flight"""
        if flight_id:
            flight = self.db.query(Flight).filter(Flight.id == flight_id).first()
        elif flight_number:
            flight = self.db.query(Flight).filter(Flight.flight_number == flight_number.upper()).first()
        else:
            return {"success": False, "error": "Flight number or ID required"}
        
        if not flight:
            return {"success": False, "error": "Flight not found"}
        
        # Get latest status update
        status_update = self.db.query(FlightStatusUpdate).filter(
            FlightStatusUpdate.flight_id == flight.id
        ).order_by(FlightStatusUpdate.updated_at.desc()).first()
        
        flight_data = {
            "success": True,
            "flight_number": flight.flight_number,
            "airline": flight.airline.name,
            "departure_airport": flight.departure_airport.code,
            "arrival_airport": flight.arrival_airport.code,
            "departure_time": flight.departure_time.isoformat(),
            "arrival_time": flight.arrival_time.isoformat(),
            "status": flight.status.value,
            "aircraft_type": flight.aircraft_type,
        }
        
        if status_update:
            flight_data["gate"] = status_update.gate
            flight_data["terminal"] = status_update.terminal
            flight_data["delay_minutes"] = status_update.delay_minutes
            flight_data["notes"] = status_update.notes
            if status_update.delay_minutes > 0:
                flight_data["estimated_departure"] = (
                    flight.departure_time.replace(minute=flight.departure_time.minute + status_update.delay_minutes)
                ).isoformat()
        
        return flight_data
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process flight status request"""
        context = context or {}
        user_lower = user_input.lower()
        
        flight_number = context.get("flight_number")
        flight_id = context.get("flight_id")
        
        if flight_number or flight_id:
            result = self.get_flight_status(flight_number=flight_number, flight_id=flight_id)
            if result["success"]:
                response_text = f"Flight Status: {result['flight_number']}\n\n"
                response_text += f"Route: {result['departure_airport']} → {result['arrival_airport']}\n"
                response_text += f"Status: {result['status'].upper()}\n"
                response_text += f"Scheduled Departure: {result['departure_time']}\n"
                response_text += f"Scheduled Arrival: {result['arrival_time']}\n"
                
                if result.get("delay_minutes", 0) > 0:
                    response_text += f"⚠️ Delay: {result['delay_minutes']} minutes\n"
                    response_text += f"Estimated Departure: {result.get('estimated_departure', 'TBD')}\n"
                
                if result.get("gate"):
                    response_text += f"Gate: {result['gate']}\n"
                if result.get("terminal"):
                    response_text += f"Terminal: {result['terminal']}\n"
                if result.get("notes"):
                    response_text += f"Notes: {result['notes']}\n"
            else:
                response_text = result.get("error", "Could not retrieve flight status")
        else:
            # General flight status information
            kb_response = self.query_knowledge_base(user_input)
            if kb_response:
                response_text = kb_response
            else:
                response_text = self.generate_response(
                    user_input,
                    "You are a flight status specialist. Provide real-time flight status information, delays, and gate details."
                )
        
        return {
            "agent": self.name,
            "response": response_text,
            "data": result if (flight_number or flight_id) else None,
            "success": True
        }


