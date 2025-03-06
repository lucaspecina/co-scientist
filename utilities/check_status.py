#!/usr/bin/env python
"""
Script to check the status of a research session.
"""

import asyncio
import json
import sys
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController

async def check_status(session_id):
    """Check the status of a research session."""
    # Initialize controller
    controller = CoScientistController(config_path="config/default_config.yaml")
    await controller.startup()
    
    # Get session status
    try:
        print(f"Checking status of session {session_id}...")
        status = await controller.get_session_status(session_id=session_id)
        print(json.dumps(status, indent=2))
    except Exception as e:
        print(f"Error checking session status: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_status.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    asyncio.run(check_status(session_id)) 