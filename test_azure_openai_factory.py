#!/usr/bin/env python
"""
Azure OpenAI Factory Test Script

This script tests the Azure OpenAI integration using the ModelFactory.
It demonstrates how to use the configuration file to create and use Azure OpenAI models.
"""

import asyncio
import os
import sys
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models.model_factory import ModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("azure_openai_factory_test")

async def test_azure_openai_factory():
    """Test the Azure OpenAI integration using the ModelFactory."""
    print("\n===== Azure OpenAI Factory Test =====\n")
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("Warning: .env file not found. Creating a template .env file...")
        with open('.env', 'w') as f:
            f.write("""# Azure OpenAI API credentials
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
""")
        print("Created .env file. Please edit it with your actual credentials and run this script again.")
        return
    
    # Load environment variables from .env file
    print("Loading environment variables from .env file...")
    load_dotenv()
    
    # Load the Azure OpenAI configuration
    config_path = "config/azure_openai_config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return
    
    # Check if environment variables are set
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    print("\nEnvironment variables from .env:")
    print(f"  AZURE_OPENAI_API_KEY: {'✅ Found' if api_key else '❌ Not set'}")
    print(f"  AZURE_OPENAI_API_VERSION: {'✅ Found' if api_version else '❌ Not set'}")
    print(f"  AZURE_OPENAI_ENDPOINT: {'✅ Found' if endpoint else '❌ Not set'}")
    
    if not all([api_key, api_version, endpoint]):
        print("\nError: Azure OpenAI environment variables are not set correctly.")
        print("Please edit your .env file and make sure the following variables are set:")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_API_VERSION")
        print("  - AZURE_OPENAI_ENDPOINT")
        return
    
    # Check if API key has placeholder text
    if api_key == "your_azure_openai_api_key_here":
        print("\nError: Your Azure OpenAI API key is still the default placeholder.")
        print("Please update your .env file with your actual Azure OpenAI API key.")
        return
        
    # Sanitize for display
    sanitized_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 10 else "[hidden]"
    
    print(f"\nUsing credentials from .env:")
    print(f"  API Key: {sanitized_key}")
    print(f"  API Version: {api_version}")
    print(f"  Endpoint: {endpoint}")
    
    # Get deployment ID (default from config)
    default_deployment = config.get("models", {}).get("azure_openai", {}).get("deployment_id", "gpt-4")
    deployment_id = input(f"\nEnter your Azure OpenAI deployment ID [default: {default_deployment}]: ").strip() or default_deployment
    
    # Update the configuration with user input
    if "models" in config and "azure_openai" in config["models"]:
        config["models"]["azure_openai"]["deployment_id"] = deployment_id
        # Add the environment variables to the config
        config["models"]["azure_openai"]["api_key"] = api_key
        config["models"]["azure_openai"]["api_version"] = api_version
        config["models"]["azure_openai"]["endpoint"] = endpoint
    
    print(f"\nCreating model with deployment: {deployment_id}")
    
    try:
        # Create the model using ModelFactory
        model_config = {
            "provider": "azure_openai",
            "deployment_id": deployment_id,
            "api_key": api_key,
            "api_version": api_version,
            "endpoint": endpoint,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        print("\nInitializing Azure OpenAI model through ModelFactory...")
        model = ModelFactory.create_model(model_config)
        
        # Test prompt
        prompt = "Explain the importance of scientific research in three sentences."
        
        print("\nSending test prompt: ", prompt)
        print("\nWaiting for response...")
        
        # Generate a response
        response = await model.generate(prompt)
        
        # Display the response
        print("\n----- Response from Azure OpenAI through ModelFactory -----")
        print(response)
        print("-------------------------------------")
        
        # Test fallback mechanism
        print("\n\n===== Testing Fallback Mechanism =====\n")
        print("This test will intentionally create a model with fallbacks.")
        print("If Azure OpenAI fails, it will try other providers.")
        
        try:
            # Create a test config with invalid Azure settings to trigger fallback
            test_config = {
                "provider": "azure_openai",
                "deployment_id": "invalid-deployment",
                "api_key": "invalid-key",
                "api_version": api_version,
                "endpoint": endpoint,
                "temperature": 0.7,
                "max_tokens": 1000,
                "default_provider": "openai"  # Fallback to OpenAI
            }
            
            print("\nTrying to create model with invalid settings (to test fallbacks)...")
            fallback_model, provider_used = ModelFactory.create_with_fallbacks(
                test_config,
                fallback_providers=["openai", "ollama"]
            )
            
            print(f"\nSuccessfully created fallback model using provider: {provider_used}")
            
            # We won't actually use the fallback model to avoid unnecessary API calls
            
        except Exception as e:
            print(f"\nFallback test failed: {e}")
            print("This might indicate that none of your fallback providers are configured correctly.")
        
        print("\n✅ Azure OpenAI factory test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error testing Azure OpenAI through ModelFactory: {str(e)}")
        print("\nPlease check your configuration and try again.")
        print("Common issues:")
        print("  - Incorrect API key or endpoint")
        print("  - Invalid deployment ID")
        print("  - API version not supported")
        print("  - Network connectivity issues")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_azure_openai_factory()) 