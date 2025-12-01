"""
Database connection and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL - supports both local and Azure PostgreSQL, with SQLite fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./airlines.db"  # SQLite fallback for easier local development
)

# For Azure, you would use:
# DATABASE_URL = os.getenv("AZURE_POSTGRESQL_CONNECTIONSTRING")

# Configure engine based on database type
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # Needed for SQLite
        echo=False
    )
else:
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,  # Better for serverless/Azure
        echo=False,
        connect_args={
            "connect_timeout": 10,
            "sslmode": "require" if "azure" in DATABASE_URL.lower() else "prefer"
        }
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    from .models import Base
    Base.metadata.create_all(bind=engine)

