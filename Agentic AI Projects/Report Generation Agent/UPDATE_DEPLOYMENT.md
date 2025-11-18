# How to Update Deployment Name

## Quick Fix

1. **Find your deployment name in Azure OpenAI Studio:**
   - Go to: https://oai.azure.com/
   - Select resource: `edukrondemoagentiaiexmp-resource`
   - Click "Deployments" in left sidebar
   - Copy the exact deployment name

2. **Update your `.env` file:**
   - Open `.env` file
   - Change this line:
     ```
     AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
     ```
   - To your actual deployment name, for example:
     ```
     AZURE_OPENAI_DEPLOYMENT_NAME=your-actual-deployment-name
     ```

3. **Common deployment names to try:**
   - `gpt-4o`
   - `gpt-4o-mini`
   - `gpt-35-turbo`
   - `gpt-4-turbo`
   - Or a custom name you created

4. **Test the connection:**
   ```bash
   python test_azure_connection.py
   ```

## Current Configuration

Your `.env` file currently has:
- ✅ API Key: Set correctly
- ✅ Endpoint: `https://edukrondemoagentiaiexmp-resource.openai.azure.com`
- ❌ Deployment Name: `gpt-4` (needs to be updated)

Once you update the deployment name, the framework will work correctly!

