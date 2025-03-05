#!/usr/bin/env python3
"""
Example of running a full research session with the AI Co-Scientist system.

This example demonstrates how to:
1. Initialize all components of the system
2. Start a research session with the supervisor
3. Monitor the progress of the session
4. Retrieve and display the final results
"""

import asyncio
import os
import sys
import logging
import yaml
import json
from datetime import datetime

# Add the project root to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.model_factory import ModelFactory
from core.memory.memory_manager import create_memory_manager
from framework.queue.task_queue import create_task_queue
from framework.supervisor.supervisor import Supervisor
from core.agents.agent_factory import create_agents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def run_full_session(
    research_goal,
    domain,
    config_path=None,
    output_dir=None,
    max_duration_minutes=30
):
    """
    Run a full research session with the AI Co-Scientist system.
    
    Args:
        research_goal (str): The research goal to explore
        domain (str): The scientific domain of the research
        config_path (str, optional): Path to a custom configuration file
        output_dir (str, optional): Directory to save session results
        max_duration_minutes (int): Maximum duration for the session in minutes
    
    Returns:
        dict: The session results
    """
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output",
            f"session_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "default_config.yaml"
        )
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize memory manager
    logger.info("Initializing memory manager")
    memory_manager = create_memory_manager(config)
    
    # Initialize task queue
    logger.info("Initializing task queue")
    task_queue = create_task_queue(config)
    
    # Initialize agents
    logger.info("Initializing agents")
    agents = create_agents(config, ModelFactory)
    
    # Initialize supervisor
    logger.info("Initializing supervisor")
    supervisor = Supervisor.from_config(
        config=config,
        agents=agents,
        memory_manager=memory_manager,
        task_queue=task_queue
    )
    
    # Define session update callback
    def session_update_callback(update):
        logger.info(f"Session update: {update.get('status')}")
        if update.get('current_task'):
            logger.info(f"Current task: {update.get('current_task').get('type')}")
        
        # Save update to file
        update_path = os.path.join(output_dir, "updates.jsonl")
        with open(update_path, "a") as f:
            f.write(json.dumps(update) + "\n")
    
    # Start session
    logger.info(f"Starting research session for goal: {research_goal}")
    session_id = await supervisor.start_session(
        research_goal=research_goal,
        domain=domain,
        callback=session_update_callback,
        max_duration_minutes=max_duration_minutes
    )
    
    logger.info(f"Session {session_id} started")
    
    # Wait for session to complete
    try:
        result = await supervisor.wait_for_session(session_id)
        
        # Save final results
        result_path = os.path.join(output_dir, "results.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print(f"Research Session Complete: {research_goal}")
        print("="*50)
        
        print("\nTop Hypotheses:")
        for i, hypothesis in enumerate(result.get("top_hypotheses", []), 1):
            print(f"\n{i}. {hypothesis.get('title')}")
            print(f"   Score: {hypothesis.get('score')}")
            print(f"   {hypothesis.get('description')}")
        
        print("\nMeta-Review:")
        print(result.get("meta_review", "No meta-review available"))
        
        print(f"\nFull results saved to: {result_path}")
        
        return result
    
    except asyncio.TimeoutError:
        logger.warning(f"Session {session_id} timed out after {max_duration_minutes} minutes")
        return {"status": "timeout", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error in session {session_id}: {str(e)}")
        return {"status": "error", "session_id": session_id, "error": str(e)}

async def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a full research session")
    parser.add_argument(
        "--goal", 
        type=str, 
        default="Identify novel drug targets for Alzheimer's disease",
        help="Research goal to explore"
    )
    parser.add_argument(
        "--domain", 
        type=str, 
        default="neuroscience",
        help="Scientific domain of the research"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to a custom configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory to save session results"
    )
    parser.add_argument(
        "--max-duration", 
        type=int, 
        default=30,
        help="Maximum duration for the session in minutes"
    )
    
    args = parser.parse_args()
    
    await run_full_session(
        research_goal=args.goal,
        domain=args.domain,
        config_path=args.config,
        output_dir=args.output_dir,
        max_duration_minutes=args.max_duration
    )

if __name__ == "__main__":
    asyncio.run(main()) 