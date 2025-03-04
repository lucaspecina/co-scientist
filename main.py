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
from typing import Dict, Any, List, Optional

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
        create_parser.add_argument("--goal", type=str, required=True, help="Research goal description")
        create_parser.add_argument("--domain", type=str, required=True, help="Scientific domain")
        create_parser.add_argument("--background", type=str, help="Background information")
        create_parser.add_argument("--constraints", type=str, nargs="+", help="Research constraints")
        
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
        
        # List sessions command
        subparsers.add_parser("list", help="List all sessions")
        
        # System info command
        subparsers.add_parser("info", help="Get system information")
        
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
        self.controller = CoScientistController(config_path=config_path)
        
        # Process command
        if self.args.command == "start":
            await self._cmd_start()
        elif self.args.command == "create":
            await self._cmd_create()
        elif self.args.command == "run":
            await self._cmd_run()
        elif self.args.command == "feedback":
            await self._cmd_feedback()
        elif self.args.command == "status":
            await self._cmd_status()
        elif self.args.command == "hypotheses":
            await self._cmd_hypotheses()
        elif self.args.command == "list":
            await self._cmd_list()
        elif self.args.command == "info":
            self._cmd_info()
    
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
        
        # Create session
        try:
            session_id = await self.controller.create_session(
                goal_description=goal,
                domain=domain,
                background=background,
                constraints=constraints
            )
            print(f"Session created successfully!")
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
    
    async def _cmd_list(self):
        """List all sessions."""
        # Ensure system is ready
        await self.controller.startup()
        
        # List sessions
        try:
            sessions = await self.controller.list_sessions()
            
            if not sessions:
                print("No sessions found.")
                return
                
            print(f"Found {len(sessions)} sessions:")
            
            for idx, session in enumerate(sessions, 1):
                print(f"\n{idx}. Session ID: {session.get('id')}")
                print(f"   Goal: {session.get('goal')}")
                print(f"   State: {session.get('state')}")
                print(f"   Iterations: {session.get('iterations_completed')}/{session.get('max_iterations')}")
            
        except Exception as e:
            print(f"Error listing sessions: {str(e)}")
            sys.exit(1)
    
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
    
    def _print_json(self, data):
        """Print data as formatted JSON."""
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    cli = CoScientistCLI()
    asyncio.run(cli.run()) 