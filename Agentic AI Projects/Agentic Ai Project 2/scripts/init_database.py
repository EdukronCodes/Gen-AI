"""
Initialize and seed the database
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal, init_db
from database.seed_data import seed_all


def main():
    """Initialize and seed database"""
    print("Initializing database...")
    init_db()
    print("✓ Database tables created")
    
    print("\nSeeding database with sample data...")
    db = SessionLocal()
    try:
        seed_all(db)
        print("\n✓ Database initialization complete!")
    except Exception as e:
        print(f"\n⚠ Error seeding database: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()


