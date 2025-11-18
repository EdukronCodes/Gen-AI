"""
Comprehensive Azure OpenAI setup checker
"""
import os
from dotenv import load_dotenv
import openai
import requests

load_dotenv()

def check_setup():
    """Check Azure OpenAI setup comprehensively"""
    
    print("=" * 70)
    print("Azure OpenAI Setup Diagnostic")
    print("=" * 70)
    
    # Get configuration
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    print(f"\n📋 Current Configuration:")
    print(f"   API Key: {'✅ Set' if api_key else '❌ Not Set'}")
    if api_key:
        print(f"   API Key (first 10 chars): {api_key[:10]}...")
    print(f"   Endpoint: {api_base}")
    print(f"   Deployment Name: {deployment_name}")
    print(f"   API Version: {api_version}")
    
    if not all([api_key, api_base, deployment_name]):
        print("\n❌ Missing required configuration!")
        return
    
    # Clean endpoint
    api_base_clean = api_base.rstrip('/')
    if api_base_clean.endswith('/openai'):
        api_base_clean = api_base_clean[:-7]
    
    print(f"\n🔧 Cleaned Endpoint: {api_base_clean}")
    
    # Test 1: Try to list deployments using REST API
    print("\n" + "=" * 70)
    print("Test 1: Listing Deployments via REST API")
    print("=" * 70)
    
    # Try different API versions
    api_versions = [
        "2024-08-01-preview",
        "2024-02-15-preview",
        "2023-12-01-preview",
        "2023-05-15"
    ]
    
    deployments_found = False
    for version in api_versions:
        url = f"{api_base_clean}/openai/deployments?api-version={version}"
        headers = {"api-key": api_key}
        
        print(f"\n   Trying API version: {version}")
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
                    deployments_found = True
                    break
                else:
                    print("   ⚠️  No deployments in response")
            elif response.status_code == 401:
                print("   ❌ Authentication failed - check your API key")
                break
            elif response.status_code == 404:
                print("   ⚠️  Endpoint not found with this API version")
            else:
                print(f"   ⚠️  Unexpected status: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {str(e)}")
    
    if not deployments_found:
        print("\n" + "=" * 70)
        print("⚠️  Could not list deployments automatically")
        print("=" * 70)
        print("\n💡 Manual Steps to Find Deployment Name:")
        print("   1. Go to: https://oai.azure.com/")
        print("   2. Select your resource")
        print("   3. Click 'Deployments' in the left sidebar")
        print("   4. You'll see a list of deployments with their names")
        print("   5. Copy the exact deployment name (case-sensitive)")
        print("   6. Update your .env file:")
        print(f"      AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name")
        print("\n   Common deployment names:")
        print("   - gpt-4")
        print("   - gpt-4-turbo")
        print("   - gpt-35-turbo")
        print("   - gpt-4o")
        print("   - gpt-4o-mini")
    
    # Test 2: Try the current deployment name
    print("\n" + "=" * 70)
    print(f"Test 2: Testing Current Deployment Name: '{deployment_name}'")
    print("=" * 70)
    
    try:
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base_clean
        )
        
        print(f"\n   Testing deployment: {deployment_name}")
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print(f"   ✅ SUCCESS! Deployment '{deployment_name}' works!")
        print(f"   Response: {response.choices[0].message.content}")
    except openai.NotFoundError as e:
        print(f"   ❌ Deployment '{deployment_name}' not found")
        print(f"   Error: {str(e)}")
        print("\n   💡 The deployment name is incorrect. Please check Azure OpenAI Studio.")
    except openai.AuthenticationError as e:
        print(f"   ❌ Authentication failed")
        print(f"   Error: {str(e)}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        print(f"   Error Type: {type(e).__name__}")

if __name__ == "__main__":
    check_setup()

