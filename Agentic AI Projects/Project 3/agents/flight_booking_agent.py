"""
Flight Booking Agent - Handles flight reservations
"""
from typing import Dict, Any
from sqlalchemy.orm import Session
from .base_agent import BaseAgent
from database.models import Flight, Passenger, Booking, SeatClass, BookingStatus
import random
import string


class FlightBookingAgent(BaseAgent):
    """Agent specialized in booking flights"""
    
    def __init__(self, db: Session):
        super().__init__(
            name="Flight Booking Agent",
            description="an expert at booking flights and creating reservations for passengers",
            db=db
        )
    
    def get_knowledge_documents(self) -> list[str]:
        return [
            "Flight booking requires passenger information, flight selection, and seat class choice.",
            "Booking references are unique 6-character alphanumeric codes.",
            "Seat classes available: economy, premium economy, business, and first class.",
            "Bookings can be in status: pending, confirmed, checked_in, cancelled, or completed.",
            "Baggage allowance varies by seat class and airline policy.",
        ]
    
    def create_booking(
        self,
        passenger_email: str,
        flight_id: int,
        seat_class: str,
        special_requests: str = None
    ) -> Dict[str, Any]:
        """Create a new flight booking"""
        # Find passenger
        passenger = self.db.query(Passenger).filter(
            Passenger.email == passenger_email
        ).first()
        
        if not passenger:
            return {
                "success": False,
                "error": f"Passenger with email {passenger_email} not found"
            }
        
        # Find flight
        flight = self.db.query(Flight).filter(Flight.id == flight_id).first()
        if not flight:
            return {
                "success": False,
                "error": f"Flight with ID {flight_id} not found"
            }
        
        # Get price based on seat class
        seat_class_enum = SeatClass[seat_class.upper()]
        price_map = {
            SeatClass.ECONOMY: flight.economy_price,
            SeatClass.PREMIUM_ECONOMY: flight.premium_economy_price,
            SeatClass.BUSINESS: flight.business_price,
            SeatClass.FIRST: flight.first_price,
        }
        
        price = price_map.get(seat_class_enum)
        if not price:
            return {
                "success": False,
                "error": f"Invalid seat class: {seat_class}"
            }
        
        # Generate booking reference
        booking_reference = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        
        # Create booking
        booking = Booking(
            booking_reference=booking_reference,
            passenger_id=passenger.id,
            flight_id=flight.id,
            seat_class=seat_class_enum,
            price=price,
            status=BookingStatus.CONFIRMED,
            baggage_allowance="23kg" if seat_class_enum == SeatClass.ECONOMY else "30kg",
            special_requests=special_requests
        )
        
        self.db.add(booking)
        self.db.commit()
        self.db.refresh(booking)
        
        return {
            "success": True,
            "booking_reference": booking_reference,
            "booking_id": booking.id,
            "price": price,
            "status": booking.status.value
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process booking request"""
        context = context or {}
        
        passenger_email = context.get("passenger_email")
        flight_id = context.get("flight_id")
        seat_class = context.get("seat_class", "economy")
        special_requests = context.get("special_requests")
        
        if not passenger_email or not flight_id:
            return {
                "agent": self.name,
                "response": "I need passenger email and flight ID to create a booking. Please provide these details.",
                "success": False
            }
        
        booking_result = self.create_booking(
            passenger_email=passenger_email,
            flight_id=flight_id,
            seat_class=seat_class,
            special_requests=special_requests
        )
        
        if booking_result["success"]:
            response_text = f"✅ Booking confirmed!\n\n"
            response_text += f"Booking Reference: {booking_result['booking_reference']}\n"
            response_text += f"Price: ${booking_result['price']}\n"
            response_text += f"Status: {booking_result['status']}\n"
            response_text += f"\nPlease save your booking reference for check-in."
        else:
            response_text = f"❌ Booking failed: {booking_result.get('error', 'Unknown error')}"
        
        return {
            "agent": self.name,
            "response": response_text,
            "data": booking_result,
            "success": booking_result["success"]
        }


