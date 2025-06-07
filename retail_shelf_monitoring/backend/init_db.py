from sqlalchemy import create_engine
from models.database import Base
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shelf_monitoring.db")
engine = create_engine(DATABASE_URL)

if __name__ == "__main__":
    print(f"Creating tables in database: {DATABASE_URL}")
    Base.metadata.create_all(engine)
    print("Database tables created successfully.") 