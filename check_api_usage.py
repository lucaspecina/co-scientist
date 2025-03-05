#!/usr/bin/env python
"""
AI Co-Scientist API Usage Checker

This script checks the API usage for a specific session.
"""

import argparse
import asyncio
import logging
import sys
from typing import Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('co_scientist.log')
    ]
)

logger = logging.getLogger("api_usage_checker")


async def check_api_usage(session_id: str):
    """Check API usage for a specific session."""
    print(f"Checking API usage for session: {session_id}")
    
    controller = CoScientistController()
    await controller.startup()
    
    try:
        # Get session data
        session_data = await controller.memory_manager.get_session(session_id)
        
        if not session_data:
            print(f"Session not found: {session_id}")
            return
        
        # Get token usage
        token_usage = session_data.get("token_usage", 0)
        
        # Estimate cost (approximate based on GPT-4 pricing)
        # Adjust these rates based on the actual models you're using
        estimated_cost = (token_usage / 1000) * 0.03  # $0.03 per 1K tokens (approximate)
        
        print("\n" + "="*50)
        print(f"API USAGE FOR SESSION: {session_id}")
        print("="*50)
        print(f"Total Tokens Used: {token_usage:,}")
        print(f"Estimated Cost: ${estimated_cost:.4f}")
        
        # Get token usage breakdown by agent if available
        agent_usage = session_data.get("agent_token_usage", {})
        if agent_usage:
            print("\nToken Usage by Agent:")
            print("-"*30)
            
            # Sort agents by token usage (descending)
            sorted_agents = sorted(
                agent_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for agent, tokens in sorted_agents:
                agent_cost = (tokens / 1000) * 0.03  # $0.03 per 1K tokens
                print(f"  {agent}: {tokens:,} tokens (${agent_cost:.4f})")
        
        # Get token usage by model if available
        model_usage = session_data.get("model_token_usage", {})
        if model_usage:
            print("\nToken Usage by Model:")
            print("-"*30)
            
            # Sort models by token usage (descending)
            sorted_models = sorted(
                model_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for model, tokens in sorted_models:
                # Adjust cost calculation based on model
                if "gpt-4" in model.lower():
                    model_cost = (tokens / 1000) * 0.03  # $0.03 per 1K tokens for GPT-4
                elif "gpt-3.5" in model.lower():
                    model_cost = (tokens / 1000) * 0.002  # $0.002 per 1K tokens for GPT-3.5
                else:
                    model_cost = (tokens / 1000) * 0.01  # Default rate
                
                print(f"  {model}: {tokens:,} tokens (${model_cost:.4f})")
        
        # Get usage over time if available
        usage_history = session_data.get("usage_history", [])
        if usage_history:
            print("\nUsage Over Time:")
            print("-"*30)
            
            for entry in usage_history:
                timestamp = entry.get("timestamp", "Unknown")
                tokens = entry.get("tokens", 0)
                print(f"  {timestamp}: {tokens:,} tokens")
        
        print("\n" + "="*50)
    
    except Exception as e:
        print(f"Error checking API usage: {str(e)}")
    finally:
        await controller.shutdown()


async def main():
    """Main entry point for the script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="AI Co-Scientist API Usage Checker"
    )
    parser.add_argument("session_id", type=str, help="Session ID to check")
    
    args = parser.parse_args()
    
    # Check API usage
    await check_api_usage(args.session_id)


if __name__ == "__main__":
    asyncio.run(main()) 