#!/usr/bin/env python
"""
AI Co-Scientist Command Line Interface

This script provides a simple command-line interface to interact with the
AI Co-Scientist system, allowing users to create and manage research sessions.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import glob
from typing import Dict, Any, List, Optional
from datetime import datetime

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

logger = logging.getLogger("co_scientist_cli")


class CoScientistCLI:
    """Command-line interface for the AI Co-Scientist system."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.controller = None
        self.args = None
    
    def parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="AI Co-Scientist: An AI system for scientific hypothesis generation and experimentation"
        )
        
        # Configuration options
        parser.add_argument("--config", type=str, help="Path to configuration file")
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Start command
        start_parser = subparsers.add_parser("start", help="Start the system")
        
        # Create session command
        create_parser = subparsers.add_parser("create", help="Create a new research session")
        create_parser.add_argument("--goal", type=str, help="Research goal description")
        create_parser.add_argument("--domain", type=str, help="Scientific domain")
        create_parser.add_argument("--background", type=str, help="Background information")
        create_parser.add_argument("--constraints", type=str, nargs="+", help="Research constraints")
        create_parser.add_argument("--json", type=str, help="Path to JSON file with research parameters")
        
        # Create session command
        create_session_parser = subparsers.add_parser("create-session", help="Create a new research session")
        create_session_parser.add_argument("--goal", type=str, required=True, help="Research goal description")
        create_session_parser.add_argument("--domain", type=str, required=True, help="Scientific domain")
        create_session_parser.add_argument("--background", type=str, help="Background information")
        create_session_parser.add_argument("--constraints", type=str, nargs="+", help="Research constraints")
        
        # Run session command
        run_parser = subparsers.add_parser("run", help="Run a research session")
        run_parser.add_argument("--session", type=str, required=True, help="Session ID")
        run_parser.add_argument("--wait", action="store_true", help="Wait for completion")
        
        # Add feedback command
        feedback_parser = subparsers.add_parser("feedback", help="Add feedback to a session")
        feedback_parser.add_argument("--session", type=str, required=True, help="Session ID")
        feedback_parser.add_argument("--text", type=str, required=True, help="Feedback text")
        feedback_parser.add_argument("--hypotheses", type=str, nargs="+", help="Target hypothesis IDs")
        
        # Get session status command
        status_parser = subparsers.add_parser("status", help="Get session status")
        status_parser.add_argument("--session", type=str, required=True, help="Session ID")
        
        # Get hypotheses command
        hypotheses_parser = subparsers.add_parser("hypotheses", help="Get hypotheses from a session")
        hypotheses_parser.add_argument("--session", type=str, required=True, help="Session ID")
        hypotheses_parser.add_argument("--limit", type=int, default=10, help="Maximum number of hypotheses")
        hypotheses_parser.add_argument("--all", action="store_true", help="Include all iterations")
        
        # Get top hypotheses command
        top_hypotheses_parser = subparsers.add_parser("top-hypotheses", help="Get top-ranked hypotheses from a session")
        top_hypotheses_parser.add_argument("--session", type=str, required=True, help="Session ID")
        top_hypotheses_parser.add_argument("--limit", type=int, default=10, help="Maximum number of hypotheses")
        
        # List sessions command
        subparsers.add_parser("list", help="List all sessions")
        
        # System info command
        subparsers.add_parser("info", help="Get system information")
        
        # Quick list sessions command
        subparsers.add_parser("quick-list", help="Quickly list all sessions")
        
        # Check API usage command
        check_usage_parser = subparsers.add_parser("check-usage", help="Check API usage for a session")
        check_usage_parser.add_argument("--session", type=str, required=True, help="Session ID to check usage for")
        
        # Delete all sessions command
        subparsers.add_parser("delete-sessions", help="Delete all sessions")
        
        # Check feedback command
        check_feedback_parser = subparsers.add_parser("check-feedback", help="Check feedback for a session")
        check_feedback_parser.add_argument("--session", type=str, required=True, help="Session ID to check feedback for")
        
        # Add Azure-specific argument
        parser.add_argument('--azure', action='store_true', help='Run the system with Azure OpenAI configuration')
        
        # Stop session command
        stop_session_parser = subparsers.add_parser("stop-session", help="Stop a specific session")
        stop_session_parser.add_argument("--session", type=str, required=True, help="Session ID to stop")
        
        # Stop all sessions command
        subparsers.add_parser("stop-all-sessions", help="Stop all active sessions")
        
        # Examine session command
        examine_session_parser = subparsers.add_parser("examine-session", help="Examine detailed session information")
        examine_session_parser.add_argument("--session", type=str, required=True, help="Session ID to examine")
        
        # Session monitor GUI command
        subparsers.add_parser("session-monitor-gui", help="Launch the session monitor GUI")
        
        # Parse arguments
        self.args = parser.parse_args()
        
        if not self.args.command:
            parser.print_help()
            sys.exit(1)
    
    async def run(self):
        """Run the CLI command."""
        # Parse arguments
        self.parse_arguments()
        
        # Initialize controller
        config_path = self.args.config
        if config_path is None:
            config_path = "config/default_config.yaml"
            
        self.controller = CoScientistController(config_path=config_path)
        
        # Azure-specific initialization
        if self.args.azure:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

            if not all([api_key, api_version, endpoint, deployment_name]):
                print("Error: Missing required Azure OpenAI environment variables.")
                sys.exit(1)

            azure_config = {
                "api_key": api_key,
                "api_version": api_version,
                "endpoint": endpoint,
                "deployment_id": deployment_name
            }

            self.controller = CoScientistController(config={"models": {"azure_openai": azure_config}})
        
        # Process command
        if self.args.command == "start":
            await self._cmd_start()
        elif self.args.command == "create":
            await self._cmd_create()
        elif self.args.command == "create-session":
            await self._cmd_create_session()
        elif self.args.command == "run":
            await self._cmd_run()
        elif self.args.command == "feedback":
            await self._cmd_feedback()
        elif self.args.command == "status":
            await self._cmd_status()
        elif self.args.command == "hypotheses":
            await self._cmd_hypotheses()
        elif self.args.command == "top-hypotheses":
            await self._cmd_top_hypotheses()
        elif self.args.command == "list":
            await self._cmd_list()
        elif self.args.command == "info":
            self._cmd_info()
        elif self.args.command == "quick-list":
            await self._cmd_quick_list()
        elif self.args.command == "check-usage":
            await self._cmd_check_usage()
        elif self.args.command == "delete-sessions":
            await self._cmd_delete_sessions()
        elif self.args.command == "check-feedback":
            await self._cmd_check_feedback()
        elif self.args.command == "stop-session":
            await self._cmd_stop_session()
        elif self.args.command == "stop-all-sessions":
            await self._cmd_stop_all_sessions()
        elif self.args.command == "examine-session":
            await self._cmd_examine_session()
        elif self.args.command == "session-monitor-gui":
            await self._cmd_session_monitor_gui()
    
    async def _cmd_start(self):
        """Start the system."""
        print("Starting AI Co-Scientist system...")
        try:
            await self.controller.startup()
            print("System started successfully!")
        except Exception as e:
            print(f"Error starting system: {str(e)}")
            sys.exit(1)
    
    async def _cmd_create(self):
        """Create a new research session."""
        print("Creating new research session...")
        
        # Ensure system is ready
        await self.controller.startup()
        
        # Get arguments
        goal = self.args.goal
        domain = self.args.domain
        background = self.args.background or ""
        constraints = self.args.constraints or []
        
        # Check if JSON file is provided
        if self.args.json:
            try:
                with open(self.args.json, 'r') as f:
                    params = json.load(f)
                    goal = params.get('goal', goal)
                    domain = params.get('domain', domain)
                    background = params.get('background', background)
                    constraints = params.get('constraints', constraints)
            except Exception as e:
                print(f"Error loading JSON file: {str(e)}")
                sys.exit(1)
        
        # Validate required arguments
        if not goal:
            print("Error: Research goal is required")
            sys.exit(1)
        if not domain:
            print("Error: Scientific domain is required")
            sys.exit(1)
        
        # Create session
        try:
            session_id = await self.controller.create_session(
                goal_description=goal,
                domain=domain,
                background=background,
                constraints=constraints
            )
            print("Session created successfully!")
            print(f"Session ID: {session_id}")
            print(f"Run the session with: python main.py run --session {session_id}")
        except Exception as e:
            print(f"Error creating session: {str(e)}")
            sys.exit(1)
    
    async def _cmd_create_session(self):
        """Create a new research session."""
        print("Creating new research session...")
        
        # Ensure system is ready
        await self.controller.startup()
        
        # Get arguments
        goal = self.args.goal
        domain = self.args.domain
        background = self.args.background or ""
        constraints = self.args.constraints or []
        
        # Validate required arguments
        if not goal:
            print("Error: Research goal is required")
            sys.exit(1)
        if not domain:
            print("Error: Scientific domain is required")
            sys.exit(1)
        
        # Create session
        try:
            session_id = await self.controller.create_session(
                goal_description=goal,
                domain=domain,
                background=background,
                constraints=constraints
            )
            print("Session created successfully!")
            print(f"Session ID: {session_id}")
            print(f"Run the session with: python main.py run --session {session_id}")
        except Exception as e:
            print(f"Error creating session: {str(e)}")
            sys.exit(1)
    
    async def _cmd_run(self):
        """Run a research session."""
        print("Running research session...")
        
        # Ensure system is ready
        await self.controller.startup()
        
        # Get arguments
        session_id = self.args.session
        wait = self.args.wait
        
        # Define status callback
        async def status_callback(session_id, old_state, new_state):
            print(f"Session state changed: {old_state} -> {new_state}")
        
        # Run session
        try:
            status = await self.controller.run_session(
                session_id=session_id,
                wait_for_completion=wait,
                status_callback=status_callback
            )
            
            print("\nSession status:")
            self._print_json(status)
            
            print("\nCurrent state:", status.get("state", "unknown"))
            
            if status.get("state") == "awaiting_feedback":
                print("\nThe session is waiting for your feedback.")
                print("Provide feedback with: python main.py feedback --session", session_id, "--text \"Your feedback here\"")
                
            if wait and status.get("state") == "completed":
                # Get hypotheses
                hypotheses = await self.controller.get_hypotheses(session_id, limit=3)
                print("\nTop hypotheses:")
                self._print_json(hypotheses)
                
        except Exception as e:
            print(f"Error running session: {str(e)}")
            sys.exit(1)
    
    async def _cmd_feedback(self):
        """Add feedback to a session."""
        print("Adding feedback to session...")
        
        # Ensure system is ready
        await self.controller.startup()
        
        # Get arguments
        session_id = self.args.session
        feedback_text = self.args.text
        target_hypothesis_ids = self.args.hypotheses
        
        # Add feedback
        try:
            status = await self.controller.add_feedback(
                session_id=session_id,
                feedback=feedback_text,
                target_hypothesis_ids=target_hypothesis_ids
            )
            
            print("Feedback added successfully!")
            print("\nSession status:")
            self._print_json(status)
            
        except Exception as e:
            print(f"Error adding feedback: {str(e)}")
            sys.exit(1)
    
    async def _cmd_status(self):
        """Get session status."""
        # Ensure system is ready
        await self.controller.startup()
        
        # Get arguments
        session_id = self.args.session
        
        # Get status
        try:
            status = await self.controller.get_session_status(session_id)
            
            print("Session status:")
            self._print_json(status)
            
        except Exception as e:
            print(f"Error getting session status: {str(e)}")
            sys.exit(1)
    
    async def _cmd_hypotheses(self):
        """Get hypotheses from a session."""
        # Ensure system is ready
        await self.controller.startup()
        
        # Get arguments
        session_id = self.args.session
        limit = self.args.limit
        include_all = self.args.all
        
        # Get hypotheses
        try:
            hypotheses = await self.controller.get_hypotheses(
                session_id=session_id,
                limit=limit,
                include_all=include_all
            )
            
            print(f"Hypotheses from session {session_id}:")
            self._print_json(hypotheses)
            
        except Exception as e:
            print(f"Error getting hypotheses: {str(e)}")
            sys.exit(1)
    
    async def _cmd_top_hypotheses(self):
        """Get top-ranked hypotheses from a session."""
        print("Getting top hypotheses...")
        
        # Ensure system is ready
        await self.controller.startup()
        
        # Get arguments
        session_id = self.args.session
        limit = self.args.limit
        
        # Get hypotheses
        try:
            hypotheses = await self.controller.get_hypotheses(
                session_id=session_id,
                limit=limit,
                include_all=False  # Only get the top-ranked hypotheses
            )
            
            print(f"\nTop {limit} hypotheses from session {session_id}:")
            print("-" * 80)
            
            if not hypotheses:
                print("No hypotheses found.")
                return
            
            for i, hypothesis in enumerate(hypotheses, 1):
                print(f"\n{i}. Score: {hypothesis.get('score', 'N/A')}")
                print(f"   Text: {hypothesis.get('text', 'N/A')}")
                print(f"   Confidence: {hypothesis.get('confidence', 'N/A')}")
                print(f"   Status: {hypothesis.get('status', 'N/A')}")
                print("-" * 80)
            
        except Exception as e:
            print(f"Error getting top hypotheses: {str(e)}")
            sys.exit(1)
    
    async def _cmd_list(self):
        """List all sessions."""
        # Using the same direct filesystem approach as quick-list
        await self._cmd_quick_list()
    
    def _cmd_info(self):
        """Get system information."""
        # Get system info
        try:
            info = self.controller.get_system_info()
            
            print("AI Co-Scientist System Information:")
            self._print_json(info)
            
        except Exception as e:
            print(f"Error getting system information: {str(e)}")
            sys.exit(1)
    
    async def _cmd_quick_list(self):
        """Quickly list all sessions."""
        print("Listing all sessions directly from file system...")
        
        # Check both possible session storage locations
        session_files = []
        session_files.extend(glob.glob("data/sessions/*.json"))
        session_files.extend(glob.glob("data/session_*.json"))
        
        # Check if data directory exists
        if not os.path.exists("data"):
            print("Data directory not found. Make sure you're in the correct project directory.")
            return
        
        if not session_files:
            print("No session files found.")
            return
        
        print(f"\nFound {len(session_files)} sessions:")
        print("-" * 100)
        print(f"{'Session ID':<30} {'Goal':<35} {'Status':<15} {'Created':<20}")
        print("-" * 100)
        
        for file_path in session_files:
            try:
                # Extract session ID from filename (without .json extension)
                filename = os.path.basename(file_path)
                if filename.startswith("session_"):
                    session_id = filename[8:].replace(".json", "")
                else:
                    session_id = filename.replace(".json", "")
                
                # Try to read the file, with detailed error reporting
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        
                    # Print file content for debugging if it's very short
                    if len(file_content) < 10:
                        print(f"{session_id:<30} {'(Empty or invalid file)':<35} {'unknown':<15} {'N/A':<20}")
                        continue
                    
                    try:
                        session = json.loads(file_content)
                        
                        # Get session details - handle different possible formats
                        goal_text = "No goal specified"
                        created_at = "Unknown"
                        
                        # First check if there's an ID field
                        if "id" in session:
                            session_id = session["id"]
                        
                        # Try different possible goal formats
                        if "goal" in session:
                            goal = session["goal"]
                            if isinstance(goal, dict) and "description" in goal:
                                # Handle case where goal is a dictionary with description
                                goal_text = goal["description"]
                            elif isinstance(goal, str):
                                # Handle case where goal is a string
                                goal_text = goal
                        
                        # Try different possible keys for the status/state
                        status = "unknown"
                        for key in ["state", "status"]:
                            if key in session and session[key]:
                                status = session[key]
                                break
                        
                        # Try to get creation time
                        for time_key in ["started_at", "created_at", "timestamp"]:
                            if time_key in session and session[time_key]:
                                try:
                                    # Parse ISO format datetime
                                    dt = datetime.fromisoformat(session[time_key].replace('Z', '+00:00'))
                                    created_at = dt.strftime("%Y-%m-%d %H:%M")
                                except:
                                    created_at = str(session[time_key])[:19]
                                break
                        
                        # Truncate goal if it's too long
                        if len(goal_text) > 32:
                            goal_text = goal_text[:29] + "..."
                        
                        print(f"{session_id:<30} {goal_text:<35} {status:<15} {created_at:<20}")
                        
                    except json.JSONDecodeError:
                        print(f"{session_id:<30} {'(Invalid JSON format)':<35} {'error':<15} {'N/A':<20}")
                        
                except UnicodeDecodeError:
                    print(f"{session_id:<30} {'(Binary or non-UTF-8 file)':<35} {'unknown':<15} {'N/A':<20}")
                    
                except Exception as e:
                    print(f"{session_id:<30} {'(Error reading file)':<35} {str(e)[:15]} {'N/A':<20}")
                    
            except Exception as e:
                # If any error occurs for a file, just show the error
                print(f"{os.path.basename(file_path):<30} {'(Error processing)':<35} {str(e)[:15]} {'N/A':<20}")
        
        print("-" * 100)
        print("\nCommands to manage sessions:")
        print("  View raw content:  type data\\session_<sessionid>.json")
        print("  Examine session:   python examine_session.py <sessionid>")
        print("  Create test session:  .\\New-BlankSession.ps1 <sessionid>")
    
    async def _cmd_check_usage(self):
        """Check API usage for a session."""
        print("Checking API usage...")
        
        # Get arguments
        session_id = self.args.session
        file_path = f"data/session_{session_id}.json"
        
        if not os.path.exists(file_path):
            file_path = f"data/sessions/{session_id}.json"
            if not os.path.exists(file_path):
                print(f"Error: Session file not found for ID: {session_id}")
                return
        
        try:
            # Read the session data
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Look for usage information in different possible locations
            usage_info = None
            if 'usage' in session_data:
                usage_info = session_data['usage']
            elif 'api_usage' in session_data:
                usage_info = session_data['api_usage']
            elif 'token_usage' in session_data:
                usage_info = {'total_tokens': session_data['token_usage']}
            
            if usage_info:
                print("API Usage for session:", session_id)
                self._print_json(usage_info)
            else:
                print("No API usage information available for this session.")
            
        except Exception as e:
            print(f"Error checking API usage: {str(e)}")
            sys.exit(1)
    
    async def _cmd_delete_sessions(self):
        """Delete all sessions."""
        print("Preparing to delete all sessions...")
        
        # Check both possible session storage locations
        session_files = []
        session_files.extend(glob.glob("data/sessions/*.json"))
        session_files.extend(glob.glob("data/session_*.json"))
        
        if not session_files:
            print("No session files found.")
            return
        
        print(f"Found {len(session_files)} session files.")
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete all {len(session_files)} sessions? (y/N): ")
        if confirm.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return
        
        # Delete the files
        deleted_count = 0
        error_count = 0
        
        for file_path in session_files:
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"  - Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  - Error deleting {os.path.basename(file_path)}: {str(e)}")
                error_count += 1
        
        print("\nSummary:")
        print(f"  - Sessions deleted: {deleted_count}")
        print(f"  - Errors encountered: {error_count}")
        print(f"  - Total sessions processed: {len(session_files)}")
    
    async def _cmd_check_feedback(self):
        """Check feedback for a session."""
        # Ensure system is ready
        await self.controller.startup()
        
        # Get arguments
        session_id = self.args.session
        
        # Check feedback
        try:
            feedback = await self.controller.check_feedback(session_id)
            
            print("Feedback for session:", session_id)
            self._print_json(feedback)
            
        except Exception as e:
            print(f"Error checking feedback: {str(e)}")
            sys.exit(1)
    
    async def _cmd_stop_session(self):
        """Stop a specific session."""
        print("Stopping session...")
        
        # Get arguments
        session_id = self.args.session
        file_path = f"data/session_{session_id}.json"
        
        if not os.path.exists(file_path):
            file_path = f"data/sessions/{session_id}.json"
            if not os.path.exists(file_path):
                print(f"Error: Session file not found for ID: {session_id}")
                return
        
        try:
            # Read the current session data
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Check current state
            current_state = None
            for state_key in ["state", "status"]:
                if state_key in session_data:
                    current_state = session_data[state_key]
                    state_field = state_key
                    break
            
            if current_state is None:
                print(f"Error: Could not determine session state")
                return
            
            # Only modify active sessions
            active_states = ["active", "evolving", "running", "initial"]
            if current_state.lower() in active_states:
                # Update the state to stopped
                session_data[state_field] = "stopped"
                
                # Add stopped_at timestamp if not present
                if "stopped_at" not in session_data:
                    session_data["stopped_at"] = datetime.now().isoformat()
                
                # Write the updated session data back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2)
                
                print(f"Session {session_id} has been stopped.")
            else:
                print(f"Session is not running (current state: {current_state})")
            
        except Exception as e:
            print(f"Error stopping session: {str(e)}")
            sys.exit(1)
    
    async def _cmd_stop_all_sessions(self):
        """Stop all active sessions."""
        print("Stopping all active sessions directly...")
        
        # Check both possible session storage locations
        session_files = []
        session_files.extend(glob.glob("data/sessions/*.json"))
        session_files.extend(glob.glob("data/session_*.json"))
        
        if not session_files:
            print("No session files found.")
            return
        
        stopped_count = 0
        already_stopped_count = 0
        error_count = 0
        
        for file_path in session_files:
            try:
                # Extract session ID for reporting
                filename = os.path.basename(file_path)
                if filename.startswith("session_"):
                    session_id = filename[8:].replace(".json", "")
                else:
                    session_id = filename.replace(".json", "")
                
                # Read the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        session_data = json.load(f)
                        
                        # Check if it's an active session that needs stopping
                        current_state = None
                        for state_key in ["state", "status"]:
                            if state_key in session_data:
                                current_state = session_data[state_key]
                                state_field = state_key
                                break
                        
                        if current_state is None:
                            print(f"  - {session_id}: Unknown state structure (no state/status field)")
                            error_count += 1
                            continue
                        
                        # Only modify active sessions
                        active_states = ["active", "evolving", "running", "initial"]
                        if current_state.lower() in active_states:
                            # Update the state to stopped
                            session_data[state_field] = "stopped"
                            
                            # Add stopped_at timestamp if not present
                            if "stopped_at" not in session_data:
                                session_data["stopped_at"] = datetime.now().isoformat()
                            
                            # Write the updated session data back to the file
                            with open(file_path, 'w', encoding='utf-8') as f_write:
                                json.dump(session_data, f_write, indent=2)
                            
                            print(f"  - {session_id}: Successfully stopped")
                            stopped_count += 1
                        else:
                            print(f"  - {session_id}: Already inactive (state: {current_state})")
                            already_stopped_count += 1
                        
                    except json.JSONDecodeError:
                        print(f"  - {session_id}: Invalid JSON format (could not stop)")
                        error_count += 1
                        
            except Exception as e:
                print(f"  - Error processing {os.path.basename(file_path)}: {str(e)}")
                error_count += 1
        
        print("\nSummary:")
        print(f"  - Sessions stopped: {stopped_count}")
        print(f"  - Sessions already inactive: {already_stopped_count}")
        print(f"  - Sessions with errors: {error_count}")
        print(f"  - Total sessions processed: {len(session_files)}")
    
    async def _cmd_examine_session(self):
        """Examine detailed session information."""
        print("Examining session information...")
        
        # Get arguments
        session_id = self.args.session
        file_path = f"data/session_{session_id}.json"
        
        if not os.path.exists(file_path):
            file_path = f"data/sessions/{session_id}.json"
            if not os.path.exists(file_path):
                print(f"Error: Session file not found for ID: {session_id}")
                return
        
        try:
            # Read the session data
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Print basic information
            print("\nSession Information:")
            print("-" * 40)
            print(f"Session ID: {session_id}")
            print(f"State: {session_data.get('state', session_data.get('status', 'unknown'))}")
            
            # Try different possible goal formats
            goal_text = "unknown"
            if "goal" in session_data:
                goal = session_data["goal"]
                if isinstance(goal, dict) and "description" in goal:
                    goal_text = goal["description"]
                elif isinstance(goal, str):
                    goal_text = goal
            print(f"Goal: {goal_text}")
            
            print(f"Domain: {session_data.get('domain', 'unknown')}")
            
            # Print hypotheses if available
            hypotheses = session_data.get("hypotheses", [])
            print(f"Total Hypotheses: {len(hypotheses)}")
            
            print("\nDetailed Information:")
            self._print_json(session_data)
            
        except Exception as e:
            print(f"Error examining session: {str(e)}")
            sys.exit(1)
    
    async def _cmd_session_monitor_gui(self):
        """Launch the session monitor GUI."""
        print("Launching session monitor GUI...")
        
        try:
            # Import the GUI module
            from session_monitor_gui import launch_gui
            
            # Launch GUI with the controller
            launch_gui(self.controller)
            
        except ImportError:
            print("Error: session_monitor_gui module not found.")
            print("Make sure the GUI component is properly installed.")
            sys.exit(1)
        except Exception as e:
            print(f"Error launching session monitor GUI: {str(e)}")
            sys.exit(1)
    
    def _print_json(self, data):
        """Print data as formatted JSON."""
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    cli = CoScientistCLI()
    asyncio.run(cli.run()) 