"""
Seed database with initial airline data
"""
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import random
from .models import (
    Airline, Airport, Flight, Passenger, Booking, Baggage,
    FlightStatus, BookingStatus, SeatClass
)


def generate_booking_reference():
    """Generate a random booking reference"""
    return ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))


def seed_airlines(db: Session):
    """Seed airlines"""
    airlines_data = [
        {"code": "AA", "name": "American Airlines", "country": "USA"},
        {"code": "DL", "name": "Delta Air Lines", "country": "USA"},
        {"code": "UA", "name": "United Airlines", "country": "USA"},
        {"code": "BA", "name": "British Airways", "country": "UK"},
        {"code": "LH", "name": "Lufthansa", "country": "Germany"},
        {"code": "AF", "name": "Air France", "country": "France"},
        {"code": "EK", "name": "Emirates", "country": "UAE"},
        {"code": "SQ", "name": "Singapore Airlines", "country": "Singapore"},
    ]
    
    for airline_data in airlines_data:
        airline = db.query(Airline).filter(Airline.code == airline_data["code"]).first()
        if not airline:
            airline = Airline(**airline_data)
            db.add(airline)
    db.commit()
    print("✓ Seeded airlines")


def seed_airports(db: Session):
    """Seed airports"""
    airports_data = [
        {"code": "JFK", "name": "John F. Kennedy International", "city": "New York", "country": "USA", "latitude": 40.6413, "longitude": -73.7781},
        {"code": "LAX", "name": "Los Angeles International", "city": "Los Angeles", "country": "USA", "latitude": 33.9416, "longitude": -118.4085},
        {"code": "LHR", "name": "Heathrow Airport", "city": "London", "country": "UK", "latitude": 51.4700, "longitude": -0.4543},
        {"code": "CDG", "name": "Charles de Gaulle", "city": "Paris", "country": "France", "latitude": 49.0097, "longitude": 2.5479},
        {"code": "FRA", "name": "Frankfurt Airport", "city": "Frankfurt", "country": "Germany", "latitude": 50.0379, "longitude": 8.5622},
        {"code": "DXB", "name": "Dubai International", "city": "Dubai", "country": "UAE", "latitude": 25.2532, "longitude": 55.3657},
        {"code": "SIN", "name": "Singapore Changi", "city": "Singapore", "country": "Singapore", "latitude": 1.3644, "longitude": 103.9915},
        {"code": "ORD", "name": "O'Hare International", "city": "Chicago", "country": "USA", "latitude": 41.9742, "longitude": -87.9073},
        {"code": "DFW", "name": "Dallas/Fort Worth International", "city": "Dallas", "country": "USA", "latitude": 32.8998, "longitude": -97.0403},
        {"code": "ATL", "name": "Hartsfield-Jackson Atlanta International", "city": "Atlanta", "country": "USA", "latitude": 33.6407, "longitude": -84.4277},
    ]
    
    for airport_data in airports_data:
        airport = db.query(Airport).filter(Airport.code == airport_data["code"]).first()
        if not airport:
            airport = Airport(**airport_data)
            db.add(airport)
    db.commit()
    print("✓ Seeded airports")


def seed_flights(db: Session):
    """Seed flights"""
    airlines = db.query(Airline).all()
    airports = db.query(Airport).all()
    
    if not airlines or not airports:
        print("⚠ Airlines or airports not found. Please seed them first.")
        return
    
    # Generate flights for the next 30 days
    base_time = datetime.now()
    flight_numbers = ["100", "200", "300", "400", "500", "600", "700", "800"]
    
    flights_created = 0
    for day in range(30):
        for airline in airlines[:4]:  # Use first 4 airlines
            for i, flight_num in enumerate(flight_numbers[:3]):  # 3 flights per airline per day
                departure_time = base_time + timedelta(days=day, hours=8 + i * 4)
                arrival_time = departure_time + timedelta(hours=random.randint(2, 8))
                
                # Random route
                departure_airport = random.choice(airports)
                arrival_airport = random.choice([a for a in airports if a.id != departure_airport.id])
                
                flight = Flight(
                    flight_number=f"{airline.code}{flight_num}",
                    airline_id=airline.id,
                    departure_airport_id=departure_airport.id,
                    arrival_airport_id=arrival_airport.id,
                    departure_time=departure_time,
                    arrival_time=arrival_time,
                    status=random.choice(list(FlightStatus)),
                    aircraft_type=random.choice(["Boeing 737", "Boeing 777", "Airbus A320", "Airbus A350"]),
                    economy_seats=150,
                    premium_economy_seats=30,
                    business_seats=20,
                    first_seats=10,
                    economy_price=round(random.uniform(200, 800), 2),
                    premium_economy_price=round(random.uniform(600, 1200), 2),
                    business_price=round(random.uniform(1500, 3000), 2),
                    first_price=round(random.uniform(4000, 8000), 2),
                )
                db.add(flight)
                flights_created += 1
    
    db.commit()
    print(f"✓ Seeded {flights_created} flights")


def seed_passengers(db: Session):
    """Seed passengers"""
    passengers_data = [
        {"first_name": "John", "last_name": "Doe", "email": "john.doe@email.com", "phone": "+1-555-0101", "loyalty_points": 5000, "membership_tier": "gold"},
        {"first_name": "Jane", "last_name": "Smith", "email": "jane.smith@email.com", "phone": "+1-555-0102", "loyalty_points": 12000, "membership_tier": "platinum"},
        {"first_name": "Michael", "last_name": "Johnson", "email": "michael.j@email.com", "phone": "+1-555-0103", "loyalty_points": 2000, "membership_tier": "silver"},
        {"first_name": "Emily", "last_name": "Williams", "email": "emily.w@email.com", "phone": "+1-555-0104", "loyalty_points": 8000, "membership_tier": "gold"},
        {"first_name": "David", "last_name": "Brown", "email": "david.brown@email.com", "phone": "+1-555-0105", "loyalty_points": 15000, "membership_tier": "platinum"},
        {"first_name": "Sarah", "last_name": "Davis", "email": "sarah.davis@email.com", "phone": "+1-555-0106", "loyalty_points": 1000, "membership_tier": "bronze"},
        {"first_name": "Robert", "last_name": "Miller", "email": "robert.m@email.com", "phone": "+1-555-0107", "loyalty_points": 3000, "membership_tier": "silver"},
        {"first_name": "Lisa", "last_name": "Wilson", "email": "lisa.wilson@email.com", "phone": "+1-555-0108", "loyalty_points": 6000, "membership_tier": "gold"},
    ]
    
    for passenger_data in passengers_data:
        passenger = db.query(Passenger).filter(Passenger.email == passenger_data["email"]).first()
        if not passenger:
            passenger = Passenger(**passenger_data)
            db.add(passenger)
    db.commit()
    print("✓ Seeded passengers")


def seed_bookings(db: Session):
    """Seed bookings"""
    passengers = db.query(Passenger).all()
    flights = db.query(Flight).limit(50).all()  # Use first 50 flights
    
    if not passengers or not flights:
        print("⚠ Passengers or flights not found.")
        return
    
    bookings_created = 0
    for passenger in passengers:
        # Create 2-3 bookings per passenger
        for _ in range(random.randint(2, 3)):
            flight = random.choice(flights)
            seat_class = random.choice(list(SeatClass))
            
            # Get price based on seat class
            price_map = {
                SeatClass.ECONOMY: flight.economy_price,
                SeatClass.PREMIUM_ECONOMY: flight.premium_economy_price,
                SeatClass.BUSINESS: flight.business_price,
                SeatClass.FIRST: flight.first_price,
            }
            
            booking = Booking(
                booking_reference=generate_booking_reference(),
                passenger_id=passenger.id,
                flight_id=flight.id,
                seat_class=seat_class,
                seat_number=f"{random.randint(1, 30)}{random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}",
                status=random.choice(list(BookingStatus)),
                price=price_map[seat_class],
                baggage_allowance=random.choice(["23kg", "30kg", "40kg"]),
                checked_in=random.choice([True, False]),
            )
            db.add(booking)
            bookings_created += 1
    
    db.commit()
    print(f"✓ Seeded {bookings_created} bookings")


def seed_all(db: Session):
    """Seed all data"""
    print("Starting database seeding...")
    seed_airlines(db)
    seed_airports(db)
    seed_flights(db)
    seed_passengers(db)
    seed_bookings(db)
    print("✓ Database seeding completed!")


if __name__ == "__main__":
    from .database import SessionLocal, init_db
    init_db()
    db = SessionLocal()
    try:
        seed_all(db)
    finally:
        db.close()


