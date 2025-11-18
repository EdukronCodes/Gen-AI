"""
Test script to diagnose Azure OpenAI connection issues
"""
import os
from dotenv import load_dotenv
import openai

load_dotenv()

def test_azure_connection():
    """Test Azure OpenAI connection with detailed error reporting"""
    
    print("=" * 60)
    print("Azure OpenAI Connection Test")
    print("=" * 60)
    
    # Get configuration
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    print(f"\n📋 Configuration:")
    print(f"   API Key: {api_key[:10]}..." if api_key else "   API Key: NOT SET")
    print(f"   Endpoint: {api_base}")
    print(f"   Deployment: {deployment_name}")
    print(f"   API Version: {api_version}")
    
    if not all([api_key, api_base, deployment_name]):
        print("\n❌ Missing required configuration!")
        return
    
    # Clean endpoint
    api_base = api_base.rstrip('/')
    if api_base.endswith('/openai'):
        api_base = api_base[:-7]
    
    print(f"\n🔧 Cleaned Endpoint: {api_base}")
    
    # Test 1: Create client
    print("\n🧪 Test 1: Creating Azure OpenAI client...")
    try:
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
        print("   ✅ Client created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create client: {str(e)}")
        return
    
    # Test 2: Simple API call
    print("\n🧪 Test 2: Testing API call...")
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello' if you can hear me."}
            ],
            max_tokens=10
        )
        print("   ✅ API call successful!")
        print(f"   Response: {response.choices[0].message.content}")
        print("\n✅ All tests passed! Azure OpenAI is configured correctly.")
    except openai.AuthenticationError as e:
        print(f"   ❌ Authentication Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Verify your API key is correct in Azure Portal")
        print("   2. Check that the endpoint URL is correct")
        print("   3. Ensure the deployment name matches exactly")
        print("   4. Verify the API version is supported")
    except openai.NotFoundError as e:
        print(f"   ❌ Not Found Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Check that the deployment name is correct")
        print("   2. Verify the deployment exists in Azure OpenAI Studio")
        print("   3. Ensure the deployment is in the same region as your resource")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        print(f"   Error Type: {type(e).__name__}")

if __name__ == "__main__":
    test_azure_connection()

