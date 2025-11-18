"""
List deployments in Azure AI Foundry project
"""
import os
from dotenv import load_dotenv
import requests

load_dotenv()

def list_deployments():
    """List all deployments in the Azure AI Foundry project"""
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip('/')
    project_name = os.getenv("AZURE_OPENAI_PROJECT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    print("=" * 70)
    print("Azure AI Foundry - Listing Deployments")
    print("=" * 70)
    print(f"\nEndpoint: {api_base}")
    print(f"Project: {project_name}")
    print(f"API Version: {api_version}\n")
    
    # Try different API versions
    api_versions = [
        "2024-08-01-preview",
        "2024-02-15-preview",
        "2023-12-01-preview",
        "2023-05-15"
    ]
    
    for version in api_versions:
        print(f"\nTrying API version: {version}")
        
        # URL for listing deployments
        url = f"{api_base}/openai/deployments?api-version={version}"
        
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        
        # Add project header if project name is provided
        if project_name:
            headers["azure-ai-project"] = project_name
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                deployments = data.get('data', [])
                if deployments:
                    print(f"\n   ✅ Found {len(deployments)} deployment(s):")
                    print("   " + "-" * 60)
                    for dep in deployments:
                        dep_id = dep.get('id', 'Unknown')
                        model = dep.get('model', 'Unknown')
                        status = dep.get('status', 'Unknown')
                        print(f"   • Deployment Name: {dep_id}")
                        print(f"     Model: {model}")
                        print(f"     Status: {status}")
                        print()
                    
                    print("\n" + "=" * 70)
                    print("✅ SUCCESS! Update your .env file with one of these deployment names:")
                    print("=" * 70)
                    for dep in deployments:
                        print(f"   AZURE_OPENAI_DEPLOYMENT_NAME={dep.get('id')}")
                    return True
                else:
                    print("   ⚠️  No deployments found in response")
            elif response.status_code == 401:
                print("   ❌ Authentication failed - check your API key")
                print(f"   Response: {response.text[:200]}")
            elif response.status_code == 404:
                print("   ⚠️  Endpoint not found with this API version")
            else:
                print(f"   ⚠️  Status {response.status_code}")
                print(f"   Response: {response.text[:300]}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {str(e)}")
    
    print("\n" + "=" * 70)
    print("⚠️  Could not list deployments automatically")
    print("=" * 70)
    print("\n💡 Manual Steps:")
    print("   1. Go to: https://oai.azure.com/")
    print("   2. Select your project: edukrondemoagentiaiexmples")
    print("   3. Click 'Deployments' in the left sidebar")
    print("   4. Copy the exact deployment name")
    print("   5. Update your .env file:")
    print("      AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name")
    
    return False

if __name__ == "__main__":
    list_deployments()

