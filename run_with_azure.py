#!/usr/bin/env python
"""
Run the AI Co-Scientist system with Azure OpenAI using our patched model factory.
"""

import os
import asyncio
import logging
import yaml
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# First, apply our patch
from fixed_model_factory import apply_patch
apply_patch()

# Now import from core
from core.controller import CoScientistController

async def main():
    """Run the AI Co-Scientist system with Azure OpenAI."""
    print("Starting AI Co-Scientist system with Azure OpenAI...")
    
    # Load the Azure OpenAI configuration
    config_path = "config/azure_openai_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Verify we have the required environment variables
    azure_openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_openai_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    print("\nAzure OpenAI Environment Variables:")
    print(f"  AZURE_OPENAI_API_KEY: {'✅ Found' if azure_openai_key else '❌ Not found'}")
    print(f"  AZURE_OPENAI_API_VERSION: {'✅ Found' if azure_openai_version else '❌ Not found'}")
    print(f"  AZURE_OPENAI_ENDPOINT: {'✅ Found' if azure_openai_endpoint else '❌ Not found'}")
    print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: {'✅ Found' if azure_openai_deployment else '❌ Not found'}")
    
    if not all([azure_openai_key, azure_openai_version, azure_openai_endpoint, azure_openai_deployment]):
        print("\nError: Missing required Azure OpenAI environment variables.")
        return
    
    # Create the controller with the configuration
    controller = CoScientistController(config=config)
    
    try:
        # Start the system
        await controller.startup()
        print("\nSystem started successfully with Azure OpenAI!")
        
        # Keep the system running (this is just a basic example)
        print("\nPress Ctrl+C to exit...\n")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping system...")
    except Exception as e:
        print(f"\nError starting system: {str(e)}")

if __name__ == "__main__":
    # Configure logging to see more details
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main()) 