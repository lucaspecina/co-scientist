#!/usr/bin/env python
"""
Script to get the top-ranked hypotheses from a research session.
"""

import asyncio
import json
import sys
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController

async def get_top_hypotheses(session_id, limit=10):
    """Get the top-ranked hypotheses from a research session."""
    # Initialize controller
    controller = CoScientistController(config_path="config/default_config.yaml")
    await controller.startup()
    
    # Get hypotheses
    try:
        print(f"Getting top hypotheses for session {session_id}...")
        hypotheses = await controller.get_hypotheses(
            session_id=session_id,
            limit=limit,
            include_all=False
        )
        print(json.dumps(hypotheses, indent=2))
    except Exception as e:
        print(f"Error getting hypotheses: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_top_hypotheses.py <session_id> [limit]")
        sys.exit(1)
    
    session_id = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    asyncio.run(get_top_hypotheses(session_id, limit)) 