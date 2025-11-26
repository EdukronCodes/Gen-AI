"""
Quick start script for the Multi-Agent Airlines System
"""
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_environment():
    """Check if required environment variables are set"""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables")
        print("   The system will work but LLM features require an OpenAI API key.")
        print("   Set it in .env file or as environment variable.")
        print()
    
    return True

def main():
    """Main entry point"""
    print("=" * 60)
    print("  Multi-Agent Airlines System")
    print("  Starting application...")
    print("=" * 60)
    print()
    
    # Check environment
    check_environment()
    
    # Initialize database
    print("Initializing database...")
    try:
        from database.database import init_db
        from database.seed_data import seed_all
        from database.database import SessionLocal
        
        init_db()
        print("✓ Database initialized")
        
        # Seed database
        print("Seeding database with sample data...")
        db = SessionLocal()
        try:
            seed_all(db)
        except Exception as e:
            print(f"⚠️  Note: {e}")
            print("   (This is okay if data already exists)")
        finally:
            db.close()
        print("✓ Database ready")
        print()
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
        print("   Continuing anyway...")
        print()
    
    # Start server
    print("Starting FastAPI server...")
    print("  API: http://localhost:8000")
    print("  UI:  http://localhost:8000/ui")
    print("  Docs: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()

