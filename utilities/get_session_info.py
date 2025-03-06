#!/usr/bin/env python
"""
Script to get detailed information about a research session.
"""

import asyncio
import json
import sys
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController

async def get_session_info(session_id):
    """Get detailed information about a research session."""
    # Initialize controller
    controller = CoScientistController(config_path="config/default_config.yaml")
    await controller.startup()
    
    try:
        # Get session status
        status = await controller.get_session_status(session_id)
        
        # Print session details
        print(f"\nSession ID: {session_id}")
        print(f"Goal: {status.get('goal', 'N/A')}")
        print(f"Status: {status.get('state', 'N/A')}")
        
        # Print top hypotheses
        print("\nTop Hypotheses:")
        top_hypotheses = status.get('top_hypotheses', [])
        if top_hypotheses:
            for i, hypothesis in enumerate(top_hypotheses):
                print(f"\n{i+1}. ID: {hypothesis.get('id', 'N/A')}")
                print(f"   Text: {hypothesis.get('text', 'N/A')}")
                print(f"   Score: {hypothesis.get('score', 'N/A')}")
        else:
            print("No top hypotheses available.")
        
        # Get all hypotheses
        hypotheses = await controller.get_hypotheses(session_id, include_all=True)
        
        # Print all hypotheses
        print("\nAll Hypotheses:")
        if hypotheses:
            for i, hypothesis in enumerate(hypotheses):
                print(f"\n{i+1}. ID: {hypothesis.get('id', 'N/A')}")
                print(f"   Title: {hypothesis.get('title', 'N/A')}")
                print(f"   Rank: {hypothesis.get('rank', 'N/A')}")
                print(f"   Created at: {hypothesis.get('created_at', 'N/A')}")
        else:
            print("No hypotheses available.")
            
    except Exception as e:
        print(f"Error retrieving session information: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_session_info.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    asyncio.run(get_session_info(session_id)) 