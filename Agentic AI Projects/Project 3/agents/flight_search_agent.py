"""
Flight Search Agent - Searches and finds available flights
"""
from typing import Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from .base_agent import BaseAgent
from database.models import Flight, Airport, Airline, FlightStatus


class FlightSearchAgent(BaseAgent):
    """Agent specialized in searching for flights"""
    
    def __init__(self, db: Session):
        super().__init__(
            name="Flight Search Agent",
            description="an expert at finding and searching " \
            "for available flights based on user criteria",
            db=db
        )
    
    def get_knowledge_documents(self) -> list[str]:
        return [
            "Flight search involves finding available flights between airports on specific dates.",
            "Users can search by departure and arrival cities, dates, and preferred airlines.",
            "Flight statuses include: scheduled, on_time, delayed, boarding, departed, arrived, cancelled.",
            "Prices vary by seat class: economy, premium economy, business, and first class.",
        ]
    
    def search_flights(
        self,
        departure_city: str = None,
        arrival_city: str = None,
        departure_date: datetime = None,
        airline_code: str = None,
        max_price: float = None
    ) -> list[Dict[str, Any]]:
        """Search for flights based on criteria"""
        query = self.db.query(Flight).join(
            Airport, Flight.departure_airport_id == Airport.id
        ).join(
            Airport, Flight.arrival_airport_id == Airport.id
        )
        
        conditions = []
        
        if departure_city:
            conditions.append(Airport.city.ilike(f"%{departure_city}%"))
        
        if arrival_city:
            conditions.append(Airport.city.ilike(f"%{arrival_city}%"))
        
        if departure_date:
            start_date = departure_date.replace(hour=0, minute=0, second=0)
            end_date = start_date.replace(hour=23, minute=59, second=59)
            conditions.append(
                and_(Flight.departure_time >= start_date, Flight.departure_time <= end_date)
            )
        
        if airline_code:
            airline = self.db.query(Airline).filter(Airline.code == airline_code).first()
            if airline:
                conditions.append(Flight.airline_id == airline.id)
        
        if conditions:
            query = query.filter(and_(*conditions))
        
        flights = query.limit(50).all()
        
        results = []
        for flight in flights:
            flight_data = {
                "flight_number": flight.flight_number,
                "airline": flight.airline.name,
                "departure_airport": flight.departure_airport.code,
                "arrival_airport": flight.arrival_airport.code,
                "departure_time": flight.departure_time.isoformat(),
                "arrival_time": flight.arrival_time.isoformat(),
                "status": flight.status.value,
                "prices": {
                    "economy": flight.economy_price,
                    "premium_economy": flight.premium_economy_price,
                    "business": flight.business_price,
                    "first": flight.first_price,
                }
            }
            
            if max_price:
                if any(price <= max_price for price in flight_data["prices"].values() if price):
                    results.append(flight_data)
            else:
                results.append(flight_data)
        
        return results
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process flight search request"""
        context = context or {}
        
        # Extract search parameters from user input
        departure_city = context.get("departure_city") or self._extract_city(user_input, "from")
        arrival_city = context.get("arrival_city") or self._extract_city(user_input, "to")
        departure_date = context.get("departure_date")
        airline_code = context.get("airline_code")
        max_price = context.get("max_price")
        
        flights = self.search_flights(
            departure_city=departure_city,
            arrival_city=arrival_city,
            departure_date=departure_date,
            airline_code=airline_code,
            max_price=max_price
        )
        
        if flights:
            response_text = f"Found {len(flights)} flights matching your criteria:\n\n"
            for flight in flights[:5]:  # Show top 5
                response_text += f"Flight {flight['flight_number']}: {flight['departure_airport']} → {flight['arrival_airport']}\n"
                response_text += f"  Departure: {flight['departure_time']}\n"
                response_text += f"  Economy: ${flight['prices']['economy']}\n"
                response_text += f"  Status: {flight['status']}\n\n"
        else:
            response_text = "No flights found matching your criteria. Please try different search parameters."
        
        return {
            "agent": self.name,
            "response": response_text,
            "data": flights,
            "success": len(flights) > 0
        }
    
    def _extract_city(self, text: str, keyword: str) -> str:
        """Simple city extraction from text"""
        text_lower = text.lower()
        if keyword in text_lower:
            # This is a simplified extraction - in production, use NLP
            parts = text_lower.split(keyword)
            if len(parts) > 1:
                city = parts[1].strip().split()[0]
                return city.title()
        return None


