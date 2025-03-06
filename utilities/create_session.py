#!/usr/bin/env python
"""
Script to create a new research session using parameters from a JSON file.
"""

import asyncio
import json
import sys
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController

async def create_session():
    """Create a new research session."""
    # Load parameters from JSON file
    try:
        with open('research_params.json', 'r') as f:
            params = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        sys.exit(1)
    
    # Extract parameters
    goal = params.get('goal')
    domain = params.get('domain')
    background = params.get('background', '')
    constraints = params.get('constraints', [])
    
    # Validate required parameters
    if not goal:
        print("Error: Research goal is required")
        sys.exit(1)
    if not domain:
        print("Error: Scientific domain is required")
        sys.exit(1)
    
    # Initialize controller
    controller = CoScientistController(config_path="config/default_config.yaml")
    await controller.startup()
    
    # Create session
    try:
        print("Creating new research session...")
        session_id = await controller.create_session(
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

if __name__ == "__main__":
    asyncio.run(create_session()) 