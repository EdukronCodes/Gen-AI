"""
Script to find the correct deployment name by trying common names
"""
import os
from dotenv import load_dotenv
import openai

load_dotenv()

def test_deployment_name(deployment_name):
    """Test if a deployment name works"""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/')
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    if api_base.endswith('/openai'):
        api_base = api_base[:-7]
    
    try:
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True, None
    except openai.NotFoundError:
        return False, "Deployment not found"
    except Exception as e:
        return False, str(e)

def find_deployment():
    """Try common deployment names"""
    
    print("=" * 60)
    print("Finding Correct Deployment Name")
    print("=" * 60)
    print("\nTrying common deployment names...\n")
    
    # Common deployment names to try
    common_names = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-35-turbo",
        "gpt-35-turbo-16k",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-32k",
        "gpt-35-turbo-1106",
    ]
    
    found = False
    for name in common_names:
        print(f"Testing: {name}...", end=" ")
        success, error = test_deployment_name(name)
        if success:
            print("✅ FOUND!")
            print(f"\n✅ Correct deployment name: {name}")
            print(f"\nUpdate your .env file with:")
            print(f"AZURE_OPENAI_DEPLOYMENT_NAME={name}")
            found = True
            break
        else:
            print(f"❌ ({error})")
    
    if not found:
        print("\n⚠️  Could not find deployment automatically.")
        print("\n💡 Please check your deployment name in Azure OpenAI Studio:")
        print("   1. Go to https://oai.azure.com/")
        print("   2. Select your resource: edukrondemoagentiaiexmp-resource")
        print("   3. Click on 'Deployments' in the left sidebar")
        print("   4. Copy the exact deployment name and update .env file")

if __name__ == "__main__":
    find_deployment()

