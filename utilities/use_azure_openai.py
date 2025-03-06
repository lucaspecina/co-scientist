#!/usr/bin/env python
"""
Directly test Azure OpenAI connectivity.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

# Load environment variables from .env file
load_dotenv()

async def test_azure_openai():
    """Test Azure OpenAI connectivity with environment variables."""
    print("\nTesting Azure OpenAI connectivity...")
    
    # Get Azure OpenAI credentials from environment variables
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment_id = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # Check if we have the necessary credentials
    print("Azure OpenAI Environment Variables:")
    print(f"  AZURE_OPENAI_API_KEY: {'✅ Found' if api_key else '❌ Not found'}")
    print(f"  AZURE_OPENAI_API_VERSION: {'✅ Found' if api_version else '❌ Not found'}")
    print(f"  AZURE_OPENAI_ENDPOINT: {'✅ Found' if endpoint else '❌ Not found'}")
    print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {'✅ Found' if deployment_id else '❌ Not found'}")
    
    if not all([api_key, api_version, endpoint, deployment_id]):
        print("\nError: Missing required Azure OpenAI environment variables")
        return
    
    try:
        # Initialize the Azure OpenAI client
        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        # Test the client with a simple completion
        response = await client.chat.completions.create(
            model=deployment_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, Azure OpenAI!"}
            ],
            max_tokens=50
        )
        
        # Display the response
        print("\nAzure OpenAI Test Response:")
        print(f"Content: {response.choices[0].message.content}")
        print("\nAzure OpenAI is working correctly! ✅")
        
        # Recommend next steps
        print("\nNext steps:")
        print("1. Create a symbolic link or modify the application code to handle Azure OpenAI properly")
        print("2. For example, you could modify core/models/model_factory.py to handle Azure configuration")
        print("3. Or create a wrapper script that properly configures the system to use Azure OpenAI")
        
    except Exception as e:
        print(f"\nError testing Azure OpenAI: {str(e)} ❌")
        print("Please check your Azure OpenAI credentials and try again.")

if __name__ == "__main__":
    asyncio.run(test_azure_openai()) 