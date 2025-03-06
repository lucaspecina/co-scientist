#!/usr/bin/env python
"""
AI Co-Scientist Session Manager

This script provides a comprehensive interface for managing AI Co-Scientist sessions:
- List all sessions (active, stopped, completed, failed)
- View detailed information about specific sessions
- Stop specific or all active sessions
- Delete specific or all sessions
- Monitor API usage across all sessions

Use this tool to prevent overspending on API usage by managing your sessions effectively.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from tabulate import tabulate
from typing import Dict, Any, List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController
from core.memory.memory_manager import FileSystemMemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('co_scientist.log')
    ]
)

logger = logging.getLogger("session_manager")


async def get_controller() -> CoScientistController:
    """
    Create and initialize a controller with a proper memory manager.
    MongoDB is currently stubbed out, so we use FileSystem instead.
    
    Returns:
        An initialized controller instance
    """
    # Initialize a controller with file-based memory manager
    controller = CoScientistController()
    
    # Replace MongoDB memory manager with FileSystem one if needed
    try:
        await controller.startup()
        # Test if memory manager is properly initialized
        try:
            sessions = await controller.list_sessions()
            # If this succeeds, we're good to go
            return controller
        except NotImplementedError:
            # MongoDB is not implemented, switch to FileSystem
            logger.info("MongoDB memory manager is not implemented, switching to FileSystem")
            # Shutdown existing controller components where possible
            if hasattr(controller, 'model_factory'):
                controller.model_factory.clear_cache()
            if hasattr(controller, 'agent_factory'):
                controller.agent_factory.clear_cache()
            
            # Create new controller with FileSystem memory manager
            fs_memory_manager = FileSystemMemoryManager(data_dir="data")
            await fs_memory_manager.startup()
            
            # Create a new controller and set its memory manager
            new_controller = CoScientistController()
            new_controller.memory_manager = fs_memory_manager
            await new_controller.startup()
            return new_controller
    except Exception as e:
        logger.error(f"Error initializing controller: {str(e)}")
        raise


async def list_sessions(filter_state: Optional[str] = None, verbose: bool = False):
    """
    List all sessions with their details.
    
    Args:
        filter_state: Optional filter to show only sessions in a specific state
                     (active, stopped, completed, failed)
        verbose: Whether to show detailed information
    """
    print("Listing sessions...")
    
    try:
        controller = await get_controller()
        
        try:
            # Get all sessions
            sessions = await controller.list_sessions()
            
            if not sessions:
                print("No sessions found.")
                return
            
            # Filter sessions if a filter is specified
            if filter_state:
                if filter_state == "active":
                    sessions = [
                        session for session in sessions 
                        if session.get("state") not in ["completed", "failed", "stopped"]
                    ]
                else:
                    sessions = [
                        session for session in sessions 
                        if session.get("state") == filter_state
                    ]
            
            if not sessions:
                print(f"No {filter_state if filter_state else ''} sessions found.")
                return
            
            total_token_usage = sum(session.get("token_usage", 0) for session in sessions)
            estimated_total_cost = (total_token_usage / 1000) * 0.03  # $0.03 per 1K tokens (approximate)
            
            print(f"Found {len(sessions)} {filter_state if filter_state else ''} sessions.")
            print(f"Total token usage: {total_token_usage:,}")
            print(f"Estimated total cost: ${estimated_total_cost:.4f}")
            print()
            
            # Prepare table data
            table_data = []
            for session in sessions:
                session_id = session.get("id")
                goal = session.get("goal", "No goal specified")
                state = session.get("state", "unknown")
                token_usage = session.get("token_usage", 0)
                estimated_cost = (token_usage / 1000) * 0.03
                created_at = session.get("created_at", "unknown")
                
                # Truncate goal if it's too long
                if len(goal) > 50:
                    goal = goal[:47] + "..."
                
                row = [
                    session_id,
                    goal,
                    state,
                    f"{token_usage:,}",
                    f"${estimated_cost:.4f}",
                    created_at
                ]
                table_data.append(row)
            
            # Sort by state (putting active sessions first) and then by token usage (descending)
            table_data.sort(key=lambda x: (
                0 if x[2] not in ["completed", "failed", "stopped"] else 1,
                -int(x[3].replace(",", "")) if x[3].replace(",", "").isdigit() else 0
            ))
            
            # Print table
            headers = ["Session ID", "Goal", "State", "Tokens", "Est. Cost", "Created At"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # If verbose mode is enabled, print more details
            if verbose:
                for session in sessions:
                    session_id = session.get("id")
                    print(f"\nDetailed information for session {session_id}:")
                    print(json.dumps(session, indent=2))
                    
        except NotImplementedError as e:
            print("Error: Session listing is not implemented in the current memory manager.")
            print("Suggestion: Check if your storage system is properly configured.")
            print(f"Technical details: {str(e)}")
        except Exception as e:
            print(f"Error retrieving sessions: {str(e)}")
    
    except Exception as e:
        print(f"Error listing sessions: {str(e)}")
    finally:
        try:
            await controller.shutdown()
        except AttributeError as e:
            # Handle case where memory manager doesn't have shutdown method
            logger.debug(f"Memory manager shutdown not implemented: {str(e)}")
            # Perform manual cleanup where possible
            if hasattr(controller, 'model_factory'):
                controller.model_factory.clear_cache()
            if hasattr(controller, 'agent_factory'):
                controller.agent_factory.clear_cache()


async def get_session_details(session_id: str):
    """Get detailed information about a specific session."""
    print(f"Getting details for session: {session_id}")
    
    try:
        controller = await get_controller()
        
        try:
            # Get session data
            session_data = await controller.memory_manager.get_session(session_id)
            
            if not session_data:
                print(f"Session not found: {session_id}")
                return
            
            # Print session details
            print("\n" + "="*50)
            print(f"SESSION DETAILS: {session_id}")
            print("="*50)
            print(f"Goal: {session_data.get('goal', 'N/A')}")
            print(f"State: {session_data.get('state', 'N/A')}")
            print(f"Created At: {session_data.get('created_at', 'N/A')}")
            
            # Token usage info
            token_usage = session_data.get("token_usage", 0)
            estimated_cost = (token_usage / 1000) * 0.03
            print(f"Token Usage: {token_usage:,}")
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
                    agent_cost = (tokens / 1000) * 0.03
                    print(f"  {agent}: {tokens:,} tokens (${agent_cost:.4f})")
            
            # Get token usage by model if available
            model_usage = session_data.get("model_token_usage", {})
            if model_usage:
                print("\nToken Usage by Model:")
                print("-"*30)
                
                sorted_models = sorted(
                    model_usage.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for model, tokens in sorted_models:
                    # Adjust cost calculation based on model
                    if "gpt-4" in model.lower():
                        model_cost = (tokens / 1000) * 0.03
                    elif "gpt-3.5" in model.lower():
                        model_cost = (tokens / 1000) * 0.002
                    else:
                        model_cost = (tokens / 1000) * 0.01
                    
                    print(f"  {model}: {tokens:,} tokens (${model_cost:.4f})")
            
            # Print top hypotheses
            print("\nTop Hypotheses:")
            print("-"*30)
            top_hypotheses = session_data.get('top_hypotheses', [])
            if top_hypotheses:
                for i, hypothesis in enumerate(top_hypotheses):
                    print(f"\n{i+1}. ID: {hypothesis.get('id', 'N/A')}")
                    print(f"   Text: {hypothesis.get('text', 'N/A')}")
                    print(f"   Score: {hypothesis.get('score', 'N/A')}")
            else:
                print("No top hypotheses available.")
            
            print("\n" + "="*50)
        except NotImplementedError as e:
            print(f"Error: Cannot retrieve session details. {str(e)}")
            print("Suggestion: Check if your storage system is properly configured.")
        except Exception as e:
            print(f"Error retrieving session details: {str(e)}")
            
    except Exception as e:
        print(f"Error getting session details: {str(e)}")
    finally:
        try:
            await controller.shutdown()
        except AttributeError as e:
            # Handle case where memory manager doesn't have shutdown method
            logger.debug(f"Memory manager shutdown not implemented: {str(e)}")
            # Perform manual cleanup where possible
            if hasattr(controller, 'model_factory'):
                controller.model_factory.clear_cache()
            if hasattr(controller, 'agent_factory'):
                controller.agent_factory.clear_cache()


async def update_session(memory_manager, session_id: str, session_data: Dict[str, Any]) -> bool:
    """
    Update a session in the memory manager.
    This is a helper function that adapts to different memory manager implementations.
    
    Args:
        memory_manager: The memory manager instance
        session_id: The ID of the session to update
        session_data: The updated session data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Try using update_session method if it exists
        if hasattr(memory_manager, 'update_session'):
            return await memory_manager.update_session(session_id, session_data)
        
        # Otherwise fall back to save_session
        await memory_manager.save_session(session_id, session_data)
        return True
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        return False


async def stop_session(session_id: str):
    """Stop a specific session."""
    print(f"Stopping session: {session_id}")
    
    try:
        controller = await get_controller()
        
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
                success = await update_session(controller.memory_manager, session_id, session_data)
                if success:
                    print(f"Session {session_id} has been stopped.")
                else:
                    print(f"Failed to update session state: {session_id}")
            else:
                print(f"Failed to stop session: {session_id}")
        
        except NotImplementedError as e:
            print(f"Error: Cannot stop session. {str(e)}")
            print("Suggestion: Check if your storage system is properly configured.")
        except Exception as e:
            print(f"Error stopping session: {str(e)}")
    
    except Exception as e:
        print(f"Error stopping session: {str(e)}")
    finally:
        try:
            await controller.shutdown()
        except AttributeError as e:
            # Handle case where memory manager doesn't have shutdown method
            logger.debug(f"Memory manager shutdown not implemented: {str(e)}")
            # Perform manual cleanup where possible
            if hasattr(controller, 'model_factory'):
                controller.model_factory.clear_cache()
            if hasattr(controller, 'agent_factory'):
                controller.agent_factory.clear_cache()


async def stop_all_sessions():
    """Stop all active sessions."""
    print("Stopping all active sessions...")
    
    try:
        controller = await get_controller()
        
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
            
            print(f"Found {len(active_sessions)} active sessions.")
            
            # Stop each active session
            stopped_count = 0
            for session in active_sessions:
                session_id = session.get("id")
                print(f"Stopping session: {session_id}")
                
                # Update session state to stopped
                session_data = await controller.memory_manager.get_session(session_id)
                if session_data:
                    session_data["state"] = "stopped"
                    success = await update_session(controller.memory_manager, session_id, session_data)
                    if success:
                        print(f"Session {session_id} has been stopped.")
                        stopped_count += 1
                    else:
                        print(f"Failed to update session state: {session_id}")
                else:
                    print(f"Failed to stop session: {session_id}")
            
            print(f"{stopped_count} out of {len(active_sessions)} active sessions have been stopped.")
        
        except NotImplementedError as e:
            print(f"Error: Cannot stop sessions. {str(e)}")
            print("Suggestion: Check if your storage system is properly configured.")
        except Exception as e:
            print(f"Error stopping sessions: {str(e)}")
    
    except Exception as e:
        print(f"Error stopping sessions: {str(e)}")
    finally:
        try:
            await controller.shutdown()
        except AttributeError as e:
            # Handle case where memory manager doesn't have shutdown method
            logger.debug(f"Memory manager shutdown not implemented: {str(e)}")
            # Perform manual cleanup where possible
            if hasattr(controller, 'model_factory'):
                controller.model_factory.clear_cache()
            if hasattr(controller, 'agent_factory'):
                controller.agent_factory.clear_cache()


async def delete_session(session_id: str, force: bool = False):
    """
    Delete a specific session.
    
    Args:
        session_id: ID of the session to delete
        force: Force deletion even if session is active
    """
    print(f"Deleting session: {session_id}")
    
    try:
        controller = await get_controller()
        
        try:
            # Get session data
            session_data = await controller.memory_manager.get_session(session_id)
            
            if not session_data:
                print(f"Session not found: {session_id}")
                return
            
            # Check if session is active and force is not specified
            if not force and session_data.get("state") not in ["completed", "failed", "stopped"]:
                print(f"Session {session_id} is still active. Use --force to delete anyway.")
                return
            
            # Delete session
            success = await controller.memory_manager.delete_session(session_id)
            
            if success:
                print(f"Session {session_id} has been deleted.")
            else:
                print(f"Failed to delete session: {session_id}")
        
        except NotImplementedError as e:
            print(f"Error: Cannot delete session. {str(e)}")
            print("Suggestion: Check if your storage system is properly configured.")
        except Exception as e:
            print(f"Error deleting session: {str(e)}")
    
    except Exception as e:
        print(f"Error deleting session: {str(e)}")
    finally:
        try:
            await controller.shutdown()
        except AttributeError as e:
            # Handle case where memory manager doesn't have shutdown method
            logger.debug(f"Memory manager shutdown not implemented: {str(e)}")
            # Perform manual cleanup where possible
            if hasattr(controller, 'model_factory'):
                controller.model_factory.clear_cache()
            if hasattr(controller, 'agent_factory'):
                controller.agent_factory.clear_cache()


async def delete_all_sessions(force: bool = False):
    """
    Delete all sessions.
    
    Args:
        force: Force deletion even if sessions are active
    """
    print("Deleting all sessions...")
    
    try:
        controller = await get_controller()
        
        try:
            # Get all sessions
            sessions = await controller.list_sessions()
            
            if not sessions:
                print("No sessions found.")
                return
            
            # Check if there are active sessions and force is not specified
            if not force:
                active_sessions = [
                    session for session in sessions 
                    if session.get("state") not in ["completed", "failed", "stopped"]
                ]
                
                if active_sessions:
                    print(f"There are {len(active_sessions)} active sessions. Use --force to delete anyway.")
                    return
            
            # Delete each session
            deleted_count = 0
            for session in sessions:
                session_id = session.get("id")
                print(f"Deleting session: {session_id}")
                
                success = await controller.memory_manager.delete_session(session_id)
                
                if success:
                    deleted_count += 1
                    print(f"Session {session_id} has been deleted.")
                else:
                    print(f"Failed to delete session: {session_id}")
            
            print(f"{deleted_count} out of {len(sessions)} sessions have been deleted.")
        
        except NotImplementedError as e:
            print(f"Error: Cannot delete sessions. {str(e)}")
            print("Suggestion: Check if your storage system is properly configured.")
        except Exception as e:
            print(f"Error deleting sessions: {str(e)}")
    
    except Exception as e:
        print(f"Error deleting sessions: {str(e)}")
    finally:
        try:
            await controller.shutdown()
        except AttributeError as e:
            # Handle case where memory manager doesn't have shutdown method
            logger.debug(f"Memory manager shutdown not implemented: {str(e)}")
            # Perform manual cleanup where possible
            if hasattr(controller, 'model_factory'):
                controller.model_factory.clear_cache()
            if hasattr(controller, 'agent_factory'):
                controller.agent_factory.clear_cache()


async def simple_list_sessions(filter_state: Optional[str] = None):
    """
    A simplified function to list sessions with minimal processing.
    This avoids the 'unhashable type: dict' errors by directly displaying session information.
    
    Args:
        filter_state: Optional filter to show only sessions in a specific state
    """
    print("Listing sessions...")
    
    try:
        # Initialize controller
        controller = CoScientistController()
        await controller.startup()
        
        try:
            # Switch to FileSystem memory manager if needed
            try:
                # Test if MongoDB is implemented
                await controller.memory_manager.list_sessions()
            except NotImplementedError:
                # MongoDB not implemented, use FileSystem
                print("Using file-based storage system instead of MongoDB...")
                fs_memory_manager = FileSystemMemoryManager(data_dir="data")
                await fs_memory_manager.startup()
                controller.memory_manager = fs_memory_manager
            
            # Get sessions directly from memory manager
            sessions = await controller.memory_manager.list_sessions()
            
            if not sessions:
                print("No sessions found.")
                return
            
            # Simple filtering if requested
            if filter_state and filter_state != "active":
                sessions = [s for s in sessions if s.get("state") == filter_state or s.get("status") == filter_state]
            elif filter_state == "active":
                sessions = [s for s in sessions if s.get("state") not in ["completed", "failed", "stopped"] 
                           and s.get("status") not in ["completed", "failed", "stopped"]]
            
            if not sessions:
                print(f"No {filter_state if filter_state else ''} sessions found.")
                return
            
            # Print simple table with just the essential information
            print(f"\nFound {len(sessions)} sessions:")
            print("-" * 80)
            print(f"{'Session ID':<20} {'Goal':<40} {'Status':<15}")
            print("-" * 80)
            
            for session in sessions:
                session_id = session.get("id", "unknown")
                # Try different possible keys for the goal/research question
                goal = (session.get("goal") or session.get("research_goal") or 
                       session.get("research_question") or "No goal specified")
                # Try different possible keys for the status/state
                status = session.get("state") or session.get("status") or "unknown"
                
                # Truncate goal if it's too long
                if len(goal) > 37:
                    goal = goal[:34] + "..."
                
                print(f"{session_id:<20} {goal:<40} {status:<15}")
            
            print("-" * 80)
            
        except Exception as e:
            print(f"Error listing sessions: {str(e)}")
        finally:
            # Clean up resources without relying on shutdown
            if hasattr(controller, 'model_factory'):
                controller.model_factory.clear_cache()
            if hasattr(controller, 'agent_factory'):
                controller.agent_factory.clear_cache()
    
    except Exception as e:
        print(f"Error initializing controller: {str(e)}")


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="AI Co-Scientist Session Manager"
    )
    
    # Add global options
    parser.add_argument("--debug", action="store_true", 
                      help="Enable debug logging")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List sessions command
    list_parser = subparsers.add_parser("list", help="List all sessions")
    list_parser.add_argument("--filter", choices=["active", "stopped", "completed", "failed"],
                          help="Filter sessions by state")
    list_parser.add_argument("--verbose", "-v", action="store_true", 
                          help="Show detailed information")
    list_parser.add_argument("--simple", action="store_true",
                          help="Use simplified function with minimal processing")
    
    # Get session details command
    details_parser = subparsers.add_parser("details", help="Get detailed info for a session")
    details_parser.add_argument("session_id", type=str, help="Session ID to get details for")
    
    # Stop session command
    stop_parser = subparsers.add_parser("stop", help="Stop a specific session")
    stop_parser.add_argument("session_id", type=str, help="Session ID to stop")
    
    # Stop all sessions command
    subparsers.add_parser("stop-all", help="Stop all active sessions")
    
    # Delete session command
    delete_parser = subparsers.add_parser("delete", help="Delete a specific session")
    delete_parser.add_argument("session_id", type=str, help="Session ID to delete")
    delete_parser.add_argument("--force", "-f", action="store_true",
                            help="Force deletion even if session is active")
    
    # Delete all sessions command
    delete_all_parser = subparsers.add_parser("delete-all", help="Delete all sessions")
    delete_all_parser.add_argument("--force", "-f", action="store_true",
                                help="Force deletion even if sessions are active")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("session_manager").setLevel(logging.DEBUG)
        print("Debug logging enabled")
    
    # Execute command based on arguments
    if args.command == "list":
        if hasattr(args, 'simple') and args.simple:
            await simple_list_sessions(filter_state=args.filter)
        else:
            await list_sessions(filter_state=args.filter, verbose=args.verbose)
    elif args.command == "details":
        await get_session_details(args.session_id)
    elif args.command == "stop":
        await stop_session(args.session_id)
    elif args.command == "stop-all":
        await stop_all_sessions()
    elif args.command == "delete":
        await delete_session(args.session_id, force=args.force)
    elif args.command == "delete-all":
        await delete_all_sessions(force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 