"""
Simple test script to verify the API is working
"""
import requests
import time
import sys

def test_api(base_url="http://localhost:8001"):
    """Test the API endpoints"""
    print("=" * 60)
    print("Testing Multi-Agent Airlines System API")
    print("=" * 60)
    print()
    
    # Test root endpoint
    try:
        print("1. Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ OK - {data.get('message', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
        else:
            print(f"   ✗ Failed - Status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print()
    
    # Test health endpoint
    try:
        print("2. Testing health endpoint...")
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ OK - Status: {data.get('status', 'Unknown')}")
            print(f"   Database: {data.get('database', 'Unknown')}")
        else:
            print(f"   ✗ Failed - Status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print()
    
    # Test agents endpoint
    try:
        print("3. Testing agents endpoint...")
        response = requests.get(f"{base_url}/api/agents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ OK - Total Agents: {data.get('total_agents', 0)}")
            print("   Agents:")
            for agent in data.get('agents', [])[:3]:
                print(f"     - {agent.get('name', 'Unknown')}")
        else:
            print(f"   ✗ Failed - Status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print()
    
    # Test query endpoint
    try:
        print("4. Testing query endpoint...")
        response = requests.post(
            f"{base_url}/api/query",
            json={"query": "Hello, what can you help me with?"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ OK - Agent: {data.get('agent_name', 'Unknown')}")
            print(f"   Routed to: {data.get('routed_to', 'Unknown')}")
            print(f"   Response preview: {data.get('response', '')[:50]}...")
        else:
            print(f"   ✗ Failed - Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   (This might be due to missing OpenAI API key)")
    
    print()
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)
    print()
    print(f"🌐 UI: {base_url}/ui")
    print(f"📡 API Docs: {base_url}/docs")
    print()
    
    return True

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "8001"
    base_url = f"http://localhost:{port}"
    
    print(f"Testing server at {base_url}")
    print("Waiting for server to be ready...")
    print()
    
    # Wait for server
    max_attempts = 10
    for i in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/", timeout=2)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(2)
        print(f"Attempt {i+1}/{max_attempts}...")
    
    print()
    test_api(base_url)


