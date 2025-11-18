"""
Startup script for the Multi-Agent AI Framework
"""
import uvicorn
import os
import sys

def check_requirements():
    """Check if all requirements are met"""
    issues = []
    
    # Check for .env file
    if not os.path.exists('.env'):
        issues.append("❌ .env file not found. Please copy .env.example to .env and add your OPENAI_API_KEY")
    
    # Check for database
    if not os.path.exists('retail_banking.db'):
        issues.append("❌ retail_banking.db not found. Please run: python create_database.py")
    
    # Check for Gemini API key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv('GEMINI_API_KEY'):
        issues.append("❌ GEMINI_API_KEY not set in .env file")
    
    if issues:
        print("\n⚠️  Setup Issues Found:\n")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease fix these issues before starting the server.\n")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent AI Framework - Starting Server")
    print("=" * 60)
    
    if not check_requirements():
        sys.exit(1)
    
    print("\n✅ All checks passed!")
    print("\n🚀 Starting server on http://localhost:8000")
    print("📖 API docs available at http://localhost:8000/docs")
    print("🌐 Frontend: Open frontend/index.html in your browser")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")

