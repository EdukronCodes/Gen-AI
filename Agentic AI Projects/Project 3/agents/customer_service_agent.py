"""
Customer Service Agent - Handles general customer inquiries
"""
from typing import Dict, Any
from sqlalchemy.orm import Session
from .base_agent import BaseAgent
from database.models import Booking, Passenger, Flight, BookingStatus


class CustomerServiceAgent(BaseAgent):
    """Agent specialized in customer service and support"""
    
    def __init__(self, db: Session):
        super().__init__(
            name="Customer Service Agent",
            description="a helpful customer service representative who assists with general inquiries, booking modifications, and support",
            db=db
        )
    
    def get_knowledge_documents(self) -> list[str]:
        return [
            "Customer service handles general inquiries, booking modifications, cancellations, and refunds.",
            "Passengers can modify bookings up to 24 hours before departure.",
            "Cancellation policies vary by fare type and airline.",
            "Refunds are processed within 5-10 business days.",
            "Special assistance is available for passengers with disabilities or medical needs.",
            "Group bookings require special handling and may have discounts.",
        ]
    
    def get_booking_info(self, booking_reference: str) -> Dict[str, Any]:
        """Get booking information"""
        booking = self.db.query(Booking).filter(
            Booking.booking_reference == booking_reference.upper()
        ).first()
        
        if not booking:
            return {"success": False, "error": "Booking not found"}
        
        return {
            "success": True,
            "booking_reference": booking.booking_reference,
            "passenger": f"{booking.passenger.first_name} {booking.passenger.last_name}",
            "flight": booking.flight.flight_number,
            "departure": booking.flight.departure_airport.code,
            "arrival": booking.flight.arrival_airport.code,
            "departure_time": booking.flight.departure_time.isoformat(),
            "seat_class": booking.seat_class.value,
            "status": booking.status.value,
            "price": booking.price
        }
    
    def cancel_booking(self, booking_reference: str) -> Dict[str, Any]:
        """Cancel a booking"""
        booking = self.db.query(Booking).filter(
            Booking.booking_reference == booking_reference.upper()
        ).first()
        
        if not booking:
            return {"success": False, "error": "Booking not found"}
        
        booking.status = BookingStatus.CANCELLED
        self.db.commit()
        
        return {
            "success": True,
            "message": "Booking cancelled successfully",
            "refund_info": "Refund will be processed within 5-10 business days"
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process customer service request"""
        context = context or {}
        user_lower = user_input.lower()
        
        # Check for booking reference
        booking_ref = context.get("booking_reference")
        
        # Determine intent
        if "cancel" in user_lower and booking_ref:
            result = self.cancel_booking(booking_ref)
            response_text = result.get("message", "Booking cancelled")
        elif booking_ref and ("info" in user_lower or "status" in user_lower or "details" in user_lower):
            booking_info = self.get_booking_info(booking_ref)
            if booking_info.get("success"):
                response_text = f"Booking Information:\n\n"
                response_text += f"Reference: {booking_info['booking_reference']}\n"
                response_text += f"Passenger: {booking_info['passenger']}\n"
                response_text += f"Flight: {booking_info['flight']}\n"
                response_text += f"Route: {booking_info['departure']} → {booking_info['arrival']}\n"
                response_text += f"Departure: {booking_info['departure_time']}\n"
                response_text += f"Class: {booking_info['seat_class']}\n"
                response_text += f"Status: {booking_info['status']}\n"
            else:
                response_text = booking_info.get("error", "Could not retrieve booking information")
        else:
            # General inquiry - use knowledge base
            kb_response = self.query_knowledge_base(user_input)
            if kb_response:
                response_text = kb_response
            else:
                response_text = self.generate_response(
                    user_input,
                    "You are a helpful customer service agent. Provide friendly, accurate assistance."
                )
        
        return {
            "agent": self.name,
            "response": response_text,
            "success": True
        }

