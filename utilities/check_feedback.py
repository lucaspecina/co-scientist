#!/usr/bin/env python
"""
Script to check if feedback was added to a research session.
"""

import asyncio
import json
import sys
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController
from core.memory.memory_manager import create_memory_manager

async def check_feedback(session_id):
    """Check if feedback was added to a research session."""
    # Initialize controller
    controller = CoScientistController(config_path="config/default_config.yaml")
    await controller.startup()
    
    try:
        # Get memory manager directly to access raw session data
        memory_manager = create_memory_manager(controller.config)
        await memory_manager.startup()
        
        # Get raw session data
        session_data = await memory_manager.get_session(session_id)
        
        # Print session details
        print(f"\nSession ID: {session_id}")
        
        # Check for feedback
        if 'feedback_history' in session_data:
            print(f"\nFeedback found:")
            for i, feedback in enumerate(session_data['feedback_history']):
                print(f"\n{i+1}. Created at: {feedback.get('timestamp', 'N/A')}")
                print(f"   Content: {feedback.get('text', 'N/A')}")
                if 'target_hypothesis_ids' in feedback and feedback['target_hypothesis_ids']:
                    print(f"   Target hypotheses: {', '.join(feedback['target_hypothesis_ids'])}")
        else:
            print("\nNo feedback found in the session data.")
            
        # Shutdown memory manager
        await memory_manager.shutdown()
            
    except Exception as e:
        print(f"Error checking feedback: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_feedback.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    asyncio.run(check_feedback(session_id)) 