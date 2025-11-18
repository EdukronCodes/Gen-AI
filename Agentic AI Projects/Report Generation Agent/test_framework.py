"""
Test script for the Multi-Agent AI Framework
"""
import asyncio
import os
from dotenv import load_dotenv
from agents.orchestrator import OrchestratorAgent
import google.generativeai as genai

load_dotenv()

async def test_framework():
    """Test the multi-agent framework"""
    
    print("=" * 60)
    print("Testing Multi-Agent AI Framework")
    print("=" * 60)
    
    # Check Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return
    
    # Check database
    if not os.path.exists("retail_banking.db"):
        print("❌ ERROR: retail_banking.db not found")
        print("   Please run: python create_database.py")
        return
    
    print("\n✅ Environment checks passed")
    
    # Initialize orchestrator
    print("\n🤖 Initializing agents...")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-pro')
    orchestrator = OrchestratorAgent(db_path="retail_banking.db", gemini_client=gemini_model)
    
    # Test question
    test_question = "How many customers are in the database?"
    print(f"\n📝 Test Question: {test_question}")
    print("\n🔄 Processing...")
    
    try:
        result = await orchestrator.execute(test_question)
        
        if result.get("status") == "success":
            print("\n✅ SUCCESS!")
            print(f"   - Query Results: {result.get('query_results_count', 0)} records")
            print(f"   - PDF Generated: {result.get('pdf_path', 'N/A')}")
            print(f"\n📄 Summary Preview:")
            summary = result.get('summary', '')[:200]
            print(f"   {summary}...")
        else:
            print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_framework())

