#!/usr/bin/env python
"""
AI Co-Scientist Session Control

This script provides commands to stop sessions and check API usage.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional

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

logger = logging.getLogger("co_scientist_control")


async def stop_session(session_id: str):
    """Stop a specific session."""
    print(f"Stopping session: {session_id}")
    
    controller = CoScientistController()
    await controller.startup()
    
    try:
        # Get session status before stopping
        status = await controller.get_session_status(session_id)
        
        if not status:
            print(f"Session not found: {session_id}")
            return
        
        # Check if session is already stopped
        if status.get("state") in ["completed", "failed", "stopped"]:
            print(f"Session is already in {status.get('state')} state.")
            return
        
        # Update session state to stopped
        session_data = await controller.memory_manager.get_session(session_id)
        if session_data:
            session_data["state"] = "stopped"
            await controller.memory_manager.update_session(session_id, session_data)
            print(f"Session {session_id} has been stopped.")
        else:
            print(f"Failed to stop session: {session_id}")
    
    except Exception as e:
        print(f"Error stopping session: {str(e)}")
    finally:
        await controller.shutdown()


async def stop_all_sessions():
    """Stop all active sessions."""
    print("Stopping all active sessions...")
    
    controller = CoScientistController()
    await controller.startup()
    
    try:
        # Get all sessions
        sessions = await controller.list_sessions()
        
        if not sessions:
            print("No active sessions found.")
            return
        
        # Filter for active sessions
        active_sessions = [
            session for session in sessions 
            if session.get("state") not in ["completed", "failed", "stopped"]
        ]
        
        if not active_sessions:
            print("No active sessions found.")
            return
        
        print(f"Found {len(active_sessions)} active sessions.")
        
        # Stop each active session
        for session in active_sessions:
            session_id = session.get("id")
            print(f"Stopping session: {session_id}")
            
            # Update session state to stopped
            session_data = await controller.memory_manager.get_session(session_id)
            if session_data:
                session_data["state"] = "stopped"
                await controller.memory_manager.update_session(session_id, session_data)
                print(f"Session {session_id} has been stopped.")
            else:
                print(f"Failed to stop session: {session_id}")
        
        print("All active sessions have been stopped.")
    
    except Exception as e:
        print(f"Error stopping sessions: {str(e)}")
    finally:
        await controller.shutdown()


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
        
        print(f"\nAPI Usage for Session: {session_id}")
        print(f"Total Tokens Used: {token_usage}")
        print(f"Estimated Cost: ${estimated_cost:.4f}")
        
        # Get token usage breakdown by agent if available
        agent_usage = session_data.get("agent_token_usage", {})
        if agent_usage:
            print("\nToken Usage by Agent:")
            for agent, tokens in agent_usage.items():
                print(f"  {agent}: {tokens} tokens")
    
    except Exception as e:
        print(f"Error checking API usage: {str(e)}")
    finally:
        await controller.shutdown()


async def shutdown_system():
    """Shutdown the entire AI Co-Scientist system."""
    print("Shutting down AI Co-Scientist system...")
    
    # First, stop all active sessions
    await stop_all_sessions()
    
    # Then shutdown the controller
    controller = CoScientistController()
    await controller.startup()
    
    try:
        await controller.shutdown()
        print("System shutdown complete.")
    except Exception as e:
        print(f"Error during system shutdown: {str(e)}")


async def list_active_sessions():
    """List all active sessions."""
    print("Listing active sessions...")
    
    controller = CoScientistController()
    await controller.startup()
    
    try:
        # Get all sessions
        sessions = await controller.list_sessions()
        
        if not sessions:
            print("No sessions found.")
            return
        
        # Filter for active sessions
        active_sessions = [
            session for session in sessions 
            if session.get("state") not in ["completed", "failed", "stopped"]
        ]
        
        if not active_sessions:
            print("No active sessions found.")
            return
        
        print(f"Found {len(active_sessions)} active sessions:")
        
        for idx, session in enumerate(active_sessions, 1):
            session_id = session.get("id")
            goal = session.get("goal", "No goal specified")
            state = session.get("state", "unknown")
            token_usage = session.get("token_usage", 0)
            
            print(f"\n{idx}. Session ID: {session_id}")
            print(f"   Goal: {goal}")
            print(f"   State: {state}")
            print(f"   Token Usage: {token_usage}")
    
    except Exception as e:
        print(f"Error listing sessions: {str(e)}")
    finally:
        await controller.shutdown()


async def main():
    """Main entry point for the script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="AI Co-Scientist Session Control"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Stop session command
    stop_parser = subparsers.add_parser("stop", help="Stop a specific session")
    stop_parser.add_argument("session_id", type=str, help="Session ID to stop")
    
    # Stop all sessions command
    subparsers.add_parser("stop-all", help="Stop all active sessions")
    
    # Check API usage command
    usage_parser = subparsers.add_parser("usage", help="Check API usage for a session")
    usage_parser.add_argument("session_id", type=str, help="Session ID to check")
    
    # Shutdown system command
    subparsers.add_parser("shutdown", help="Shutdown the entire system")
    
    # List active sessions command
    subparsers.add_parser("list-active", help="List all active sessions")
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == "stop":
        await stop_session(args.session_id)
    elif args.command == "stop-all":
        await stop_all_sessions()
    elif args.command == "usage":
        await check_api_usage(args.session_id)
    elif args.command == "shutdown":
        await shutdown_system()
    elif args.command == "list-active":
        await list_active_sessions()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 