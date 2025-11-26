"""
Check-in Agent - Handles flight check-in processes
"""
from typing import Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from .base_agent import BaseAgent
from database.models import Booking, Flight, BookingStatus


class CheckInAgent(BaseAgent):
    """Agent specialized in flight check-in"""
    
    def __init__(self, db: Session):
        super().__init__(
            name="Check-in Agent",
            description="an expert at processing flight check-ins, seat assignments, and boarding passes",
            db=db
        )
    
    def get_knowledge_documents(self) -> list[str]:
        return [
            "Online check-in is available 24 hours before departure.",
            "Airport check-in closes 60 minutes before departure for domestic flights.",
            "International flights require check-in 90 minutes before departure.",
            "Seat selection is available during check-in.",
            "Boarding passes can be digital or printed at the airport.",
            "Passengers must have valid ID and travel documents for check-in.",
        ]
    
    def check_in(self, booking_reference: str, seat_number: str = None) -> Dict[str, Any]:
        """Process check-in for a booking"""
        booking = self.db.query(Booking).filter(
            Booking.booking_reference == booking_reference.upper()
        ).first()
        
        if not booking:
            return {"success": False, "error": "Booking not found"}
        
        if booking.checked_in:
            return {
                "success": False,
                "error": "Already checked in",
                "checked_in_at": booking.checked_in_at.isoformat() if booking.checked_in_at else None
            }
        
        # Check if check-in is allowed (within 24 hours of departure)
        flight = booking.flight
        time_until_departure = (flight.departure_time - datetime.now()).total_seconds() / 3600
        
        if time_until_departure < 0:
            return {"success": False, "error": "Flight has already departed"}
        
        if time_until_departure > 24:
            return {
                "success": False,
                "error": f"Check-in opens 24 hours before departure. Your flight departs in {int(time_until_departure)} hours."
            }
        
        # Process check-in
        booking.checked_in = True
        booking.checked_in_at = datetime.now()
        booking.status = BookingStatus.CHECKED_IN
        
        if seat_number:
            booking.seat_number = seat_number
        
        self.db.commit()
        
        return {
            "success": True,
            "booking_reference": booking_reference,
            "flight_number": flight.flight_number,
            "seat_number": booking.seat_number,
            "gate": "TBD",  # Would come from flight status
            "boarding_time": (flight.departure_time.replace(minute=flight.departure_time.minute - 30)).isoformat(),
            "departure_time": flight.departure_time.isoformat()
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process check-in request"""
        context = context or {}
        user_lower = user_input.lower()
        
        booking_ref = context.get("booking_reference")
        seat_number = context.get("seat_number")
        
        if booking_ref and ("check" in user_lower or "check-in" in user_lower):
            result = self.check_in(booking_ref, seat_number)
            if result["success"]:
                response_text = f"✅ Check-in successful!\n\n"
                response_text += f"Booking Reference: {result['booking_reference']}\n"
                response_text += f"Flight: {result['flight_number']}\n"
                response_text += f"Seat: {result['seat_number']}\n"
                response_text += f"Boarding Time: {result['boarding_time']}\n"
                response_text += f"Departure: {result['departure_time']}\n"
                response_text += f"\nYour boarding pass is ready!"
            else:
                response_text = f"❌ Check-in failed: {result.get('error', 'Unknown error')}"
        else:
            # General check-in information
            kb_response = self.query_knowledge_base(user_input)
            if kb_response:
                response_text = kb_response
            else:
                response_text = self.generate_response(
                    user_input,
                    "You are a check-in specialist. Provide information about check-in procedures, timing, and requirements."
                )
        
        return {
            "agent": self.name,
            "response": response_text,
            "success": True
        }


