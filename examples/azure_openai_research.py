#!/usr/bin/env python
"""
Example of using Azure OpenAI for AI Co-Scientist research

This example demonstrates how to:
1. Configure Azure OpenAI as the model provider
2. Initialize a generation agent with Azure OpenAI
3. Generate hypotheses for a research goal
"""

import asyncio
import os
import sys
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.model_factory import ModelFactory
from core.agents.generation_agent import GenerationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def generate_hypotheses_with_azure_openai(
    research_goal="Identify novel drug targets for Alzheimer's disease",
    domain="neuroscience",
    deployment_id=None,
    api_key=None,
    api_version=None,
    endpoint=None
):
    """
    Generate research hypotheses using Azure OpenAI.
    
    Args:
        research_goal (str): The research goal to generate hypotheses for
        domain (str): The scientific domain of the research
        deployment_id (str, optional): Azure OpenAI deployment ID
        api_key (str, optional): Azure OpenAI API key
        api_version (str, optional): Azure OpenAI API version
        endpoint (str, optional): Azure OpenAI endpoint
    
    Returns:
        list: The generated hypotheses
    """
    logger.info(f"Generating hypotheses using Azure OpenAI for goal: {research_goal}")
    
    # Load Azure OpenAI configuration
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "azure_openai_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override with provided values or environment variables
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
    endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    if not all([api_key, api_version, endpoint]):
        raise ValueError("Azure OpenAI credentials not provided in .env file. Please set environment variables or provide credentials as arguments.")
    
    # Update config with credentials
    config["models"]["azure_openai"]["api_key"] = api_key
    config["models"]["azure_openai"]["api_version"] = api_version
    config["models"]["azure_openai"]["endpoint"] = endpoint
    
    if deployment_id:
        config["models"]["azure_openai"]["deployment_id"] = deployment_id
    
    # Create the model using ModelFactory
    model_config = {
        "provider": "azure_openai",
        "deployment_id": config["models"]["azure_openai"]["deployment_id"],
        "api_key": api_key,
        "api_version": api_version,
        "endpoint": endpoint,
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    model = ModelFactory.create_model(model_config)
    
    # Load the generation agent template
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "templates",
        "generation_agent.txt"
    )
    with open(template_path, "r") as f:
        template = f.read()
    
    # Create the generation agent
    agent = GenerationAgent(
        model=model,
        prompt_template=template
    )
    
    # Generate hypotheses
    hypotheses = await agent.generate_hypotheses(
        goal=research_goal,
        domain=domain,
        count=3
    )
    
    return hypotheses

async def main():
    """Run the example."""
    print("\n===== Azure OpenAI Research Example =====\n")
    
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
    
    # Check if environment variables are set
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    print("\nEnvironment variables from .env:")
    print(f"  AZURE_OPENAI_API_KEY: {'✅ Found' if api_key else '❌ Not set'}")
    print(f"  AZURE_OPENAI_API_VERSION: {'✅ Found' if api_version else '❌ Not set'}")
    print(f"  AZURE_OPENAI_ENDPOINT: {'✅ Found' if endpoint else '❌ Not set'}")
    
    if not all([api_key, api_version, endpoint]):
        print("\nError: Azure OpenAI environment variables are not set correctly in your .env file.")
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
    
    # Get deployment ID from user
    deployment_id = input("\nEnter your Azure OpenAI deployment ID [default: gpt-4]: ").strip() or "gpt-4"
    
    # Get research goal from user
    research_goal = input("\nEnter your research goal [default: Identify novel drug targets for Alzheimer's disease]: ").strip()
    if not research_goal:
        research_goal = "Identify novel drug targets for Alzheimer's disease"
    
    # Get scientific domain from user
    domain = input("\nEnter scientific domain [default: neuroscience]: ").strip()
    if not domain:
        domain = "neuroscience"
    
    print(f"\nGenerating hypotheses using Azure OpenAI deployment: {deployment_id}")
    print(f"Research goal: {research_goal}")
    print(f"Domain: {domain}")
    print("\nGenerating hypotheses. Please wait...\n")
    
    try:
        hypotheses = await generate_hypotheses_with_azure_openai(
            research_goal=research_goal,
            domain=domain,
            deployment_id=deployment_id
        )
        
        print("\n===== Generated Hypotheses =====\n")
        for i, hypothesis in enumerate(hypotheses, 1):
            print(f"Hypothesis {i}: {hypothesis}")
            print("-" * 80)
        
        print("\n✅ Successfully generated hypotheses using Azure OpenAI!")
        
    except Exception as e:
        print(f"\n❌ Error generating hypotheses: {str(e)}")
        print("\nPlease check your configuration and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 