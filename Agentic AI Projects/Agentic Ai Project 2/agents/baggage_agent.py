"""
Baggage Information Agent - Handles baggage-related queries
"""
from typing import Dict, Any
from sqlalchemy.orm import Session
from .base_agent import BaseAgent
from database.models import Booking, Baggage, SeatClass


class BaggageAgent(BaseAgent):
    """Agent specialized in baggage information and policies"""
    
    def __init__(self, db: Session):
        super().__init__(
            name="Baggage Information Agent",
            description="an expert on baggage policies, allowances, restrictions, and tracking",
            db=db
        )
    
    def get_knowledge_documents(self) -> list[str]:
        return [
            "Economy class typically allows 23kg (50lbs) of checked baggage.",
            "Premium economy allows 30kg (66lbs) of checked baggage.",
            "Business class allows 40kg (88lbs) of checked baggage.",
            "First class allows 50kg (110lbs) of checked baggage.",
            "Carry-on baggage is typically limited to 7-10kg and must fit in overhead bins.",
            "Restricted items include liquids over 100ml, sharp objects, and flammable materials.",
            "Baggage tracking is available using tracking numbers provided at check-in.",
            "Excess baggage fees apply for weight or pieces exceeding allowance.",
        ]
    
    def get_baggage_allowance(self, booking_reference: str) -> Dict[str, Any]:
        """Get baggage allowance for a booking"""
        booking = self.db.query(Booking).filter(
            Booking.booking_reference == booking_reference.upper()
        ).first()
        
        if not booking:
            return {"success": False, "error": "Booking not found"}
        
        # Determine allowance based on seat class
        allowance_map = {
            SeatClass.ECONOMY: {"checked": "23kg", "carry_on": "7kg", "pieces": 1},
            SeatClass.PREMIUM_ECONOMY: {"checked": "30kg", "carry_on": "10kg", "pieces": 1},
            SeatClass.BUSINESS: {"checked": "40kg", "carry_on": "10kg", "pieces": 2},
            SeatClass.FIRST: {"checked": "50kg", "carry_on": "10kg", "pieces": 2},
        }
        
        allowance = allowance_map.get(booking.seat_class, allowance_map[SeatClass.ECONOMY])
        
        return {
            "success": True,
            "booking_reference": booking_reference,
            "seat_class": booking.seat_class.value,
            "allowance": allowance,
            "current_allowance": booking.baggage_allowance
        }
    
    def track_baggage(self, tracking_number: str) -> Dict[str, Any]:
        """Track baggage by tracking number"""
        baggage = self.db.query(Baggage).filter(
            Baggage.tracking_number == tracking_number.upper()
        ).first()
        
        if not baggage:
            return {"success": False, "error": "Tracking number not found"}
        
        return {
            "success": True,
            "tracking_number": tracking_number,
            "weight": baggage.weight_kg,
            "is_checked": baggage.is_checked,
            "status": "Checked in" if baggage.is_checked else "Not checked in"
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process baggage-related request"""
        context = context or {}
        user_lower = user_input.lower()
        
        booking_ref = context.get("booking_reference")
        tracking_number = context.get("tracking_number")
        
        if "track" in user_lower and tracking_number:
            result = self.track_baggage(tracking_number)
            if result["success"]:
                response_text = f"Baggage Tracking:\n\n"
                response_text += f"Tracking Number: {result['tracking_number']}\n"
                response_text += f"Weight: {result['weight']}kg\n"
                response_text += f"Status: {result['status']}\n"
            else:
                response_text = result.get("error", "Could not track baggage")
        elif booking_ref and ("allowance" in user_lower or "baggage" in user_lower):
            result = self.get_baggage_allowance(booking_ref)
            if result["success"]:
                response_text = f"Baggage Allowance for {result['seat_class']} class:\n\n"
                response_text += f"Checked Baggage: {result['allowance']['checked']}\n"
                response_text += f"Carry-on: {result['allowance']['carry_on']}\n"
                response_text += f"Pieces: {result['allowance']['pieces']}\n"
            else:
                response_text = result.get("error", "Could not retrieve baggage information")
        else:
            # General baggage inquiry
            kb_response = self.query_knowledge_base(user_input)
            if kb_response:
                response_text = kb_response
            else:
                response_text = self.generate_response(
                    user_input,
                    "You are a baggage information specialist. Provide detailed information about baggage policies, allowances, and restrictions."
                )
        
        return {
            "agent": self.name,
            "response": response_text,
            "success": True
        }


