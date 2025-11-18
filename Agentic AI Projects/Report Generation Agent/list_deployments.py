"""
Script to list available Azure OpenAI deployments
"""
import os
from dotenv import load_dotenv
import openai
try:
    import requests
except ImportError:
    print("⚠️  requests library not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "requests"])
    import requests

load_dotenv()

def list_deployments():
    """List all available deployments"""
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/')
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    # Remove /openai if present
    if api_base.endswith('/openai'):
        api_base = api_base[:-7]
    
    print("=" * 60)
    print("Listing Azure OpenAI Deployments")
    print("=" * 60)
    print(f"\nEndpoint: {api_base}")
    print(f"API Version: {api_version}\n")
    
    # Try to list deployments via REST API
    url = f"{api_base}/openai/deployments?api-version={api_version}"
    headers = {
        "api-key": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            deployments = data.get('data', [])
            if deployments:
                print("✅ Available Deployments:")
                print("-" * 60)
                for dep in deployments:
                    print(f"  • {dep.get('id', 'Unknown')}")
                    print(f"    Model: {dep.get('model', 'Unknown')}")
                    print(f"    Status: {dep.get('status', 'Unknown')}")
                    print()
            else:
                print("⚠️  No deployments found")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Error listing deployments: {str(e)}")
        print("\n💡 Please check your deployment name in Azure OpenAI Studio:")
        print("   1. Go to https://oai.azure.com/")
        print("   2. Select your resource")
        print("   3. Click on 'Deployments' in the left sidebar")
        print("   4. Note the exact deployment name(s)")

if __name__ == "__main__":
    list_deployments()

