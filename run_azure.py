#!/usr/bin/env python
"""
Run script that correctly initializes the Azure OpenAI configuration.
"""

import os
import asyncio
import yaml
from dotenv import load_dotenv

from core.controller import CoScientistController

# Load environment variables from .env file
load_dotenv()

async def main():
    """Run the AI Co-Scientist system with Azure OpenAI configuration."""
    print("Starting AI Co-Scientist system with Azure OpenAI...")
    
    # Load the configuration
    config_path = "config/azure_openai_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Explicitly set the Azure OpenAI credentials from environment variables
    azure_openai_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_openai_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    # Check if we have the required environment variables
    if not all([azure_openai_key, azure_openai_version, azure_openai_endpoint]):
        print("Error: Missing required Azure OpenAI environment variables:")
        if not azure_openai_key:
            print("- AZURE_OPENAI_API_KEY")
        if not azure_openai_version:
            print("- AZURE_OPENAI_API_VERSION")
        if not azure_openai_endpoint:
            print("- AZURE_OPENAI_ENDPOINT")
        return
    
    # Update the configuration with the environment variables
    if 'models' in config and 'azure_openai' in config['models']:
        config['models']['azure_openai']['api_key'] = azure_openai_key
        config['models']['azure_openai']['api_version'] = azure_openai_version
        config['models']['azure_openai']['endpoint'] = azure_openai_endpoint
        
        if azure_openai_deployment:
            config['models']['azure_openai']['deployment_id'] = azure_openai_deployment
    
    # Create the controller with the in-memory configuration
    controller = CoScientistController(config=config)
    
    # Start the system
    try:
        await controller.startup()
        print("System started successfully with Azure OpenAI!")
        
        # Keep the system running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping system...")
    except Exception as e:
        print(f"Error starting system: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 