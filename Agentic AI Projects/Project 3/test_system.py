"""
Simple test to verify the system is working
"""
import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_endpoints():
    print("=" * 60)
    print("  Testing Multi-Agent Airlines System")
    print("=" * 60)
    print()
    
    # Test root
    try:
        print("1. Testing root endpoint...")
        r = requests.get(f"{BASE_URL}/", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"   [OK] - {data.get('message')}")
            print(f"   Version: {data.get('version')}")
        else:
            print(f"   [FAILED] - Status: {r.status_code}")
    except Exception as e:
        print(f"   [ERROR] - {e}")
        return False
    
    print()
    
    # Test health
    try:
        print("2. Testing health endpoint...")
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"   [OK] - Status: {data.get('status')}")
            print(f"   Database: {data.get('database')}")
        else:
            print(f"   [FAILED] - Status: {r.status_code}")
    except Exception as e:
        print(f"   [ERROR] - {e}")
    
    print()
    
    # Test agents
    try:
        print("3. Testing agents endpoint...")
        r = requests.get(f"{BASE_URL}/api/agents", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"   [OK] - Total Agents: {data.get('total_agents')}")
            print("   Agents:")
            for agent in data.get('agents', []):
                print(f"     - {agent.get('name')}")
        else:
            print(f"   [FAILED] - Status: {r.status_code}")
    except Exception as e:
        print(f"   [ERROR] - {e}")
    
    print()
    
    # Test query
    try:
        print("4. Testing query endpoint...")
        payload = {
            "query": "Hello, what can you help me with?",
            "context": {}
        }
        r = requests.post(
            f"{BASE_URL}/api/query",
            json=payload,
            timeout=30
        )
        if r.status_code == 200:
            data = r.json()
            print(f"   [OK] - Agent: {data.get('agent_name')}")
            print(f"   Routed to: {data.get('routed_to')}")
            response_preview = data.get('response', '')[:80]
            print(f"   Response: {response_preview}...")
        else:
            print(f"   [FAILED] - Status: {r.status_code}")
            print(f"   Response: {r.text[:200]}")
    except Exception as e:
        print(f"   [ERROR] - {e}")
        print("   (This might be due to OpenAI API initialization)")
    
    print()
    print("=" * 60)
    print("  System Test Complete!")
    print("=" * 60)
    print()
    print(f"Web UI:  {BASE_URL}/ui")
    print(f"API Docs: {BASE_URL}/docs")
    print()

if __name__ == "__main__":
    print("Waiting for server to be ready...")
    time.sleep(2)
    test_endpoints()

