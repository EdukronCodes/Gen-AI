"""
Database models for Airlines Multi-Agent System
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class FlightStatus(enum.Enum):
    SCHEDULED = "scheduled"
    ON_TIME = "on_time"
    DELAYED = "delayed"
    BOARDING = "boarding"
    DEPARTED = "departed"
    ARRIVED = "arrived"
    CANCELLED = "cancelled"


class BookingStatus(enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class SeatClass(enum.Enum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class Airline(Base):
    __tablename__ = "airlines"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(3), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    flights = relationship("Flight", back_populates="airline")


class Airport(Base):
    __tablename__ = "airports"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(3), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    city = Column(String(50), nullable=False)
    country = Column(String(50), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)
    
    departures = relationship("Flight", foreign_keys="Flight.departure_airport_id", back_populates="departure_airport")
    arrivals = relationship("Flight", foreign_keys="Flight.arrival_airport_id", back_populates="arrival_airport")


class Flight(Base):
    __tablename__ = "flights"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_number = Column(String(10), nullable=False, index=True)
    airline_id = Column(Integer, ForeignKey("airlines.id"), nullable=False)
    departure_airport_id = Column(Integer, ForeignKey("airports.id"), nullable=False)
    arrival_airport_id = Column(Integer, ForeignKey("airports.id"), nullable=False)
    departure_time = Column(DateTime, nullable=False)
    arrival_time = Column(DateTime, nullable=False)
    status = Column(SQLEnum(FlightStatus), default=FlightStatus.SCHEDULED)
    aircraft_type = Column(String(50))
    economy_seats = Column(Integer, default=150)
    premium_economy_seats = Column(Integer, default=30)
    business_seats = Column(Integer, default=20)
    first_seats = Column(Integer, default=10)
    economy_price = Column(Float, nullable=False)
    premium_economy_price = Column(Float)
    business_price = Column(Float)
    first_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    airline = relationship("Airline", back_populates="flights")
    departure_airport = relationship("Airport", foreign_keys=[departure_airport_id], back_populates="departures")
    arrival_airport = relationship("Airport", foreign_keys=[arrival_airport_id], back_populates="arrivals")
    bookings = relationship("Booking", back_populates="flight")


class Passenger(Base):
    __tablename__ = "passengers"
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    phone = Column(String(20))
    date_of_birth = Column(DateTime)
    passport_number = Column(String(20))
    loyalty_points = Column(Integer, default=0)
    membership_tier = Column(String(20), default="bronze")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    bookings = relationship("Booking", back_populates="passenger")


class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    booking_reference = Column(String(10), unique=True, nullable=False, index=True)
    passenger_id = Column(Integer, ForeignKey("passengers.id"), nullable=False)
    flight_id = Column(Integer, ForeignKey("flights.id"), nullable=False)
    seat_class = Column(SQLEnum(SeatClass), nullable=False)
    seat_number = Column(String(10))
    status = Column(SQLEnum(BookingStatus), default=BookingStatus.PENDING)
    price = Column(Float, nullable=False)
    baggage_allowance = Column(String(50), default="23kg")
    special_requests = Column(String(500))
    checked_in = Column(Boolean, default=False)
    checked_in_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    passenger = relationship("Passenger", back_populates="bookings")
    flight = relationship("Flight", back_populates="bookings")


class Baggage(Base):
    __tablename__ = "baggage"
    
    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(Integer, ForeignKey("bookings.id"), nullable=False)
    weight_kg = Column(Float, nullable=False)
    dimensions = Column(String(50))
    is_checked = Column(Boolean, default=False)
    tracking_number = Column(String(20), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    booking = relationship("Booking")


class FlightStatusUpdate(Base):
    __tablename__ = "flight_status_updates"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(Integer, ForeignKey("flights.id"), nullable=False)
    status = Column(SQLEnum(FlightStatus), nullable=False)
    gate = Column(String(10))
    terminal = Column(String(10))
    delay_minutes = Column(Integer, default=0)
    notes = Column(String(500))
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    flight = relationship("Flight")


