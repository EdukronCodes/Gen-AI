"""
Azure OpenAI Client Helper
"""
import openai
import os
from dotenv import load_dotenv

load_dotenv()

def create_azure_openai_client():
    """
    Create and return an Azure OpenAI client (supports both standard Azure OpenAI and Azure AI Foundry)
    
    Required environment variables:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI/AI Foundry API key
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
    - AZURE_OPENAI_DEPLOYMENT_NAME: Your deployment name (e.g., 'gpt-4', 'gpt-35-turbo')
    - AZURE_OPENAI_API_VERSION: API version (default: '2024-08-01-preview')
    - AZURE_OPENAI_PROJECT_NAME: (Optional) Project name for Azure AI Foundry
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    project_name = os.getenv("AZURE_OPENAI_PROJECT_NAME")
    
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
    if not api_base:
        raise ValueError("AZURE_OPENAI_ENDPOINT not found in environment variables")
    if not deployment_name:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME not found in environment variables")
    
    # Clean up endpoint URL - remove trailing slash if present
    api_base = api_base.rstrip('/')
    
    # Ensure endpoint doesn't have /openai suffix (Azure SDK adds it)
    if api_base.endswith('/openai'):
        api_base = api_base[:-7]
    
    try:
        # For Azure AI Foundry, we might need to use default_headers with project
        client_kwargs = {
            "api_key": api_key,
            "api_version": api_version,
            "azure_endpoint": api_base
        }
        
        # If project name is provided (Azure AI Foundry), add it to default headers
        if project_name:
            client_kwargs["default_headers"] = {
                "azure-ai-project": project_name
            }
        
        client = openai.AzureOpenAI(**client_kwargs)
        
        # Store deployment name for later use
        client.deployment_name = deployment_name
        
        return client
    except Exception as e:
        raise ValueError(f"Failed to create Azure OpenAI client: {str(e)}. Please verify your API key, endpoint, and deployment name.")

