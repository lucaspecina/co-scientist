#!/usr/bin/env python
"""
Script to run a research session.
"""

import asyncio
import sys
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController

async def run_session(session_id):
    """Run a research session."""
    # Initialize controller
    controller = CoScientistController(config_path="config/default_config.yaml")
    await controller.startup()
    
    # Run session
    try:
        print(f"Running session {session_id}...")
        await controller.run_session(session_id=session_id, wait=True)
        print("Session completed successfully!")
    except Exception as e:
        print(f"Error running session: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_session.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    asyncio.run(run_session(session_id)) 