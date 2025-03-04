#!/usr/bin/env python3
"""
Main Entry Point for AI Co-Scientist

This script provides a command-line interface to the AI Co-Scientist system.
It demonstrates how to use the model-agnostic architecture with different
LLM providers.
"""

import argparse
import asyncio
import logging
import os
import sys
import yaml
from typing import Dict, Any, Optional

from core.models.model_factory import ModelFactory
from core.models.base_model import BaseModel
from core.agents.generation_agent import GenerationAgent
from core.memory.memory_manager import create_memory_manager
from framework.queue.task_queue import create_task_queue
from framework.supervisor.supervisor import Supervisor


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_single_generation(model_provider: str, research_goal: str, api_key: Optional[str] = None):
    """
    Run a single hypothesis generation using the specified model provider.
    This demonstrates how to use the model-agnostic design with different providers.
    
    Args:
        model_provider: Name of the model provider to use
        research_goal: Research goal to generate hypotheses for
        api_key: Optional API key for the model provider
    """
    # Load default configuration
    with open("config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Override default provider
    config["models"]["default_provider"] = model_provider
    
    # Set API key if provided
    if api_key:
        config["models"][model_provider]["api_key"] = api_key
    
    # Create model instance
    model_config = {
        "provider": model_provider,
        "default_provider": model_provider,
        model_provider: config["models"][model_provider]
    }
    model = ModelFactory.create_model(model_config)
    
    # Create generation agent
    agent_config = config["agents"]["generation"]
    agent = GenerationAgent(
        model=model,
        prompt_template_path=agent_config.get("prompt_template", "config/templates/generation_agent.txt"),
        num_hypotheses=5,
        creativity=0.7
    )
    
    # Run generation
    logger.info(f"Generating hypotheses using {model_provider} for research goal: {research_goal}")
    
    result = await agent.run_with_timing({
        "research_goal": research_goal,
        "domain": "general science"
    })
    
    # Print results
    if "error" in result:
        logger.error(f"Error generating hypotheses: {result['error']}")
    else:
        hypotheses = result.get("hypotheses", [])
        logger.info(f"Generated {len(hypotheses)} hypotheses:")
        
        for i, hypothesis in enumerate(hypotheses):
            print(f"\n--- Hypothesis {i+1} ---")
            print(f"Title: {hypothesis.get('title', 'N/A')}")
            print(f"Description: {hypothesis.get('description', 'N/A')}")
            print(f"Mechanism: {hypothesis.get('mechanism', 'N/A')}")
            print(f"Testability: {hypothesis.get('testability', 'N/A')}")
            
        if "metadata" in result:
            print(f"\nExecution time: {result['metadata'].get('execution_time', 0):.2f} seconds")


async def run_full_session(config_path: str, research_goal: str, domain: str = "general science"):
    """
    Run a complete research session with the multi-agent workflow.
    
    Args:
        config_path: Path to configuration file
        research_goal: Research goal to generate hypotheses for
        domain: Scientific domain
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize memory manager
    memory_manager = create_memory_manager(config)
    
    # Initialize task queue
    task_queue = create_task_queue(config)
    
    # Initialize agents
    agents = {}
    
    # Process each agent type
    for agent_type, agent_config in config.get("agents", {}).items():
        count = agent_config.get("count", 1)
        agents[agent_type] = []
        
        # Create the specified number of instances
        for i in range(count):
            # Get a model for this agent
            model = ModelFactory.create_model_for_agent(agent_type, config)
            
            # Create agent instance (example for generation agent)
            if agent_type == "generation":
                agent = GenerationAgent.from_config(agent_config, model)
                agents[agent_type].append(agent)
            # Add other agent types here...
            else:
                logger.warning(f"Agent type {agent_type} not implemented yet")
    
    # Initialize supervisor
    supervisor = Supervisor.from_config(config, agents, memory_manager, task_queue)
    
    # Define update callback
    def session_update_callback(session):
        status = session.get("status", "unknown")
        iteration = session.get("current_iteration", 0)
        
        if status == "completed":
            logger.info(f"Session completed after {iteration} iterations")
        elif status.endswith("_complete"):
            logger.info(f"Phase {status} completed (iteration {iteration})")
    
    # Start session
    session_id = await supervisor.start_session(
        research_goal=research_goal,
        domain=domain,
        callback=session_update_callback
    )
    
    logger.info(f"Started session {session_id}")
    
    # Wait for session to complete
    while True:
        session = await supervisor.get_session(session_id)
        if session.get("complete", False) or session.get("status") == "error":
            break
        await asyncio.sleep(1)
    
    # Get final results
    session = await supervisor.get_session(session_id)
    
    if session.get("status") == "error":
        logger.error(f"Session failed: {session.get('error', 'Unknown error')}")
        return
    
    # Print top hypotheses
    top_hypotheses = session.get("top_hypotheses", [])
    logger.info(f"Session completed with {len(top_hypotheses)} top hypotheses:")
    
    for i, hypothesis in enumerate(top_hypotheses):
        print(f"\n--- Top Hypothesis {i+1} ---")
        print(f"Title: {hypothesis.get('title', 'N/A')}")
        print(f"Description: {hypothesis.get('description', 'N/A')}")
        print(f"Mechanism: {hypothesis.get('mechanism', 'N/A')}")
        if "rank" in hypothesis:
            print(f"Rank: {hypothesis.get('rank')}")
    
    # Print meta-review
    meta_review = session.get("meta_review", {})
    if meta_review:
        print("\n--- Meta Review ---")
        print(f"Summary: {meta_review.get('summary', 'N/A')}")
        print(f"Recommendations: {meta_review.get('recommendations', 'N/A')}")


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="AI Co-Scientist CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single generation command
    gen_parser = subparsers.add_parser("generate", help="Run a single hypothesis generation")
    gen_parser.add_argument(
        "--model", "-m", 
        choices=["openai", "anthropic", "google", "huggingface", "local"],
        default="openai",
        help="Model provider to use"
    )
    gen_parser.add_argument(
        "--goal", "-g", 
        required=True,
        help="Research goal to generate hypotheses for"
    )
    gen_parser.add_argument(
        "--api-key", "-k",
        help="API key for the model provider (overrides environment variable)"
    )
    
    # Full session command
    session_parser = subparsers.add_parser("session", help="Run a complete research session")
    session_parser.add_argument(
        "--config", "-c",
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    session_parser.add_argument(
        "--goal", "-g",
        required=True,
        help="Research goal to generate hypotheses for"
    )
    session_parser.add_argument(
        "--domain", "-d",
        default="general science",
        help="Scientific domain"
    )
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "generate":
        # Run single generation
        asyncio.run(run_single_generation(args.model, args.goal, args.api_key))
    elif args.command == "session":
        # Run full session
        asyncio.run(run_full_session(args.config, args.goal, args.domain))
    elif args.command == "server":
        # Start API server
        import uvicorn
        from ui.api.server import app
        
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # No command specified
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 