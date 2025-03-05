#!/usr/bin/env python
"""
Azure OpenAI Test Script

This script tests whether your Azure OpenAI configuration is working correctly.
It sends a simple prompt to your Azure OpenAI deployment and displays the response.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models.azure_openai_model import AzureOpenAIModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("azure_openai_test")

async def list_models(api_key, api_version, endpoint):
    """List available models in Azure OpenAI."""
    try:
        import requests
        from openai import AsyncAzureOpenAI

        # First try via models.list API
        try:
            client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            
            response = await client.models.list()
            
            print("\n=== Available Models/Deployments ===")
            if hasattr(response, 'data') and response.data:
                for model in response.data:
                    print(f"- {model.id}")
                return True
        except Exception as e:
            print(f"Could not list models using OpenAI SDK: {str(e)}")
            
        # Try direct REST API call as fallback
        print("\nTrying alternative method to list deployments...")
        
        # Try with management API
        try:
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }
            
            # Try direct endpoint for deployments
            deployment_url = f"{endpoint}/openai/deployments?api-version={api_version}"
            print(f"Trying to list deployments from: {deployment_url}")
            
            response = requests.get(deployment_url, headers=headers)
            
            if response.status_code == 200:
                deployments = response.json()
                print("\n=== Available Deployments ===")
                if "data" in deployments and deployments["data"]:
                    for deployment in deployments["data"]:
                        print(f"- {deployment['id']}")
                    return True
                else:
                    print("No deployments found in the response.")
            else:
                print(f"Failed to list deployments: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error in alternative method: {str(e)}")
        
        print("\n=== IMPORTANT: Manual Deployment Check ===")
        print("1. Go to the Azure Portal: https://portal.azure.com")
        print("2. Navigate to your Azure OpenAI resource")
        print("3. Click on 'Model deployments' in the left menu")
        print("4. Use the exact deployment name shown in the 'Name' column")
        print("\nRemember: In Azure OpenAI, you must use the DEPLOYMENT NAME, not the model name")
        
        return False
        
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return False

async def test_azure_openai():
    """Test Azure OpenAI model."""
    # Check if .env file exists
    env_path = Path(".env")
    if not env_path.exists():
        print("No .env file found. Creating template...")
        with open(env_path, "w") as f:
            f.write("# Azure OpenAI API credentials\n")
            f.write("AZURE_OPENAI_API_KEY=your_api_key_here\n")
            f.write("AZURE_OPENAI_API_VERSION=2024-02-15-preview\n")
            f.write("AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/\n")
            f.write("AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name\n")
            f.write("# Optional: AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name\n")
        print(f".env file created at {env_path.absolute()}")
        print("Please fill in your API key, API version, endpoint, and deployment name.")
        return

    print("\n===== Azure OpenAI Test =====\n")
    print("Loading environment variables from .env file...")
    load_dotenv()

    # Get API key and endpoint from environment variables
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

    # Display environment variable status
    print("\nEnvironment variables:")
    print(f"  AZURE_OPENAI_API_KEY: {'✅ Found' if api_key else '❌ Not found'}")
    print(f"  AZURE_OPENAI_API_VERSION: {'✅ Found' if api_version else '❌ Not found'}")
    print(f"  AZURE_OPENAI_ENDPOINT: {'✅ Found' if endpoint else '❌ Not found'}")
    print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {'✅ Found' if deployment_name else '❌ Not found'}")

    if not api_key or not endpoint:
        print("\nPlease set your Azure OpenAI API key and endpoint in the .env file.")
        return

    # Display variables being used
    print("\nUsing variables from .env:")
    print(f"  API Key: {api_key[:5]}...{api_key[-4:]}")
    print(f"  API Version: {api_version}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Deployment Name: {deployment_name or 'Not set - will use default'}")

    # Note about model listing - no prompt
    print("\nNote: To list available models in your Azure OpenAI resource, go to:")
    print("Azure Portal > Your OpenAI resource > Model deployments")
    
    # Ask if user wants to enable debug mode
    debug_mode = input("\nEnable debug mode for detailed logs? (y/n) [default: n]: ").lower() == "y"
    
    # Get deployment ID from environment or default
    if deployment_name:
        deployment_id = deployment_name
        print(f"\nUsing deployment name from .env: {deployment_id}")
    else:
        # Use default deployment ID
        deployment_id = "gpt-4"
        print(f"\nNo deployment name found in .env, using default: {deployment_id}")
        
    # Create OpenAI model
    try:
        print("\nInitializing Azure OpenAI model...")
        model = AzureOpenAIModel(
            api_key=api_key,
            api_version=api_version,
            endpoint=endpoint,
            deployment_id=deployment_id,
            debug=debug_mode
        )
        
        # Get prompt from user
        prompt = input("\nEnter your prompt: ")
        if not prompt:
            prompt = "Explain the importance of scientific research in three sentences."
            print(f"Using default prompt: '{prompt}'")

        print("\nGenerating response...")
        response = await model.generate(prompt)
        
        print("\n=== Response ===")
        print(response)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        
        if "404" in str(e):
            print("\n=== TROUBLESHOOTING 404 ERROR ===")
            print("1. Verify your deployment ID exists in Azure OpenAI Studio")
            print("2. Check if your API version is compatible with this deployment")
            print("3. Make sure your endpoint URL is correct")
            print("4. Confirm that the deployment is fully provisioned (may take a few minutes)")
            print("\nFor Azure OpenAI, the URL format should be:")
            print(f"{endpoint}/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}")
            
        elif "401" in str(e):
            print("\n=== TROUBLESHOOTING 401 ERROR ===")
            print("Your API key may be incorrect or expired")
            
        elif "429" in str(e):
            print("\n=== TROUBLESHOOTING 429 ERROR ===")
            print("You've exceeded your rate limit or quota")
            
        print("\nFor more help, refer to Azure OpenAI documentation:")
        print("https://learn.microsoft.com/en-us/azure/ai-services/openai/")

if __name__ == "__main__":
    asyncio.run(test_azure_openai()) 