#!/usr/bin/env python3
"""
Basic example of using the AI Co-Scientist system to generate research hypotheses.

This example demonstrates how to:
1. Configure a model provider
2. Initialize a generation agent
3. Generate hypotheses for a research goal
"""

import asyncio
import os
import sys
import logging
import yaml

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

async def generate_hypotheses(
    provider="openai",
    research_goal="Identify novel drug targets for Alzheimer's disease",
    domain="neuroscience",
    api_key=None
):
    """
    Generate research hypotheses using the specified model provider.
    
    Args:
        provider (str): The model provider to use (openai, google, anthropic, local)
        research_goal (str): The research goal to generate hypotheses for
        domain (str): The scientific domain of the research
        api_key (str, optional): API key for the model provider
    
    Returns:
        list: The generated hypotheses
    """
    logger.info(f"Generating hypotheses using {provider} for goal: {research_goal}")
    
    # Load default configuration
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "default_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Override API key if provided
    if api_key:
        if provider in config["models"]:
            config["models"][provider]["api_key"] = api_key
    
    # Create model instance
    model_config = {
        "provider": provider,
        **config["models"]
    }
    model = ModelFactory.create_model(model_config)
    
    # Create generation agent
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "templates",
        "generation_agent.txt"
    )
    agent = GenerationAgent(
        model=model,
        prompt_template_path=template_path,
        num_hypotheses=3
    )
    
    # Run generation
    result = await agent.run_with_timing({
        "research_goal": research_goal,
        "domain": domain,
        "constraints": "Focus on mechanisms that can be targeted with small molecules",
        "prior_knowledge": "Recent studies have implicated neuroinflammation in Alzheimer's progression"
    })
    
    # Print results
    print("\n" + "="*50)
    print(f"Generated {len(result.get('hypotheses', []))} hypotheses:")
    print("="*50)
    
    for i, hypothesis in enumerate(result.get("hypotheses", []), 1):
        print(f"\nHypothesis {i}:")
        print(f"Title: {hypothesis.get('title')}")
        print(f"Description: {hypothesis.get('description')}")
        print(f"Mechanism: {hypothesis.get('mechanism')}")
        print(f"Testing: {hypothesis.get('testing')}")
        print(f"Novelty: {hypothesis.get('novelty')}")
        print(f"Impact: {hypothesis.get('impact')}")
        print("-"*50)
    
    return result.get("hypotheses", [])

async def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate research hypotheses")
    parser.add_argument(
        "--provider", 
        type=str, 
        default="openai",
        choices=["openai", "google", "anthropic", "local"],
        help="Model provider to use"
    )
    parser.add_argument(
        "--goal", 
        type=str, 
        default="Identify novel drug targets for Alzheimer's disease",
        help="Research goal to generate hypotheses for"
    )
    parser.add_argument(
        "--domain", 
        type=str, 
        default="neuroscience",
        help="Scientific domain of the research"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="API key for the model provider"
    )
    
    args = parser.parse_args()
    
    await generate_hypotheses(
        provider=args.provider,
        research_goal=args.goal,
        domain=args.domain,
        api_key=args.api_key
    )

if __name__ == "__main__":
    asyncio.run(main()) 