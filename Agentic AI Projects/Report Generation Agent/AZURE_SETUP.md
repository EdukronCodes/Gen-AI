# Azure OpenAI Setup Guide

## Important: Deployment Name

The `.env` file has been configured with a default deployment name of `gpt-4`. 

**You need to verify and update the `AZURE_OPENAI_DEPLOYMENT_NAME` in your `.env` file to match your actual deployment name in Azure OpenAI Studio.**

## How to Find Your Deployment Name

1. Go to [Azure OpenAI Studio](https://oai.azure.com/)
2. Select your resource: `openaiapirequest2025`
3. Navigate to **Deployments** in the left sidebar
4. You'll see a list of your deployments (e.g., "gpt-4", "gpt-35-turbo", etc.)
5. Copy the exact deployment name and update it in your `.env` file

## Current Configuration

Your `.env` file is configured with:
- **API Key**: `FUKnNalvGpZkUAiyBsh96`
- **Endpoint**: `https://openaiapirequest2025.openai.azure.com/`
- **Deployment Name**: `gpt-4` (⚠️ **Verify this matches your Azure deployment**)
- **API Version**: `2024-02-15-preview`

## Update Deployment Name

If your deployment name is different, edit the `.env` file:

```env
AZURE_OPENAI_DEPLOYMENT_NAME=your-actual-deployment-name
```

Common deployment names:
- `gpt-4`
- `gpt-4-turbo`
- `gpt-35-turbo`
- `gpt-35-turbo-16k`

## Testing

After updating the deployment name, test the connection:

```bash
python test_framework.py
```

Or start the server:

```bash
python start_server.py
```

## Troubleshooting

If you get errors about the deployment name:
1. Verify the deployment exists in Azure OpenAI Studio
2. Check that the deployment name matches exactly (case-sensitive)
3. Ensure the deployment is in the same region as your resource
4. Verify the API version is supported by your deployment

