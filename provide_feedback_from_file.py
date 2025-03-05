#!/usr/bin/env python
"""
Script to provide feedback on hypotheses in a research session from a file.
"""

import asyncio
import json
import sys
from dotenv import load_dotenv
load_dotenv()

from core.controller import CoScientistController

async def provide_feedback_from_file(session_id, feedback_file, hypothesis_ids=None):
    """Provide feedback on hypotheses in a research session from a file."""
    # Read feedback from file
    try:
        with open(feedback_file, 'r') as f:
            feedback_text = f.read()
    except Exception as e:
        print(f"Error reading feedback file: {str(e)}")
        sys.exit(1)
    
    # Initialize controller
    controller = CoScientistController(config_path="config/default_config.yaml")
    await controller.startup()
    
    # Provide feedback
    try:
        print(f"Providing feedback for session {session_id}...")
        await controller.add_feedback(
            session_id=session_id,
            feedback=feedback_text,
            target_hypothesis_ids=hypothesis_ids
        )
        print("Feedback submitted successfully!")
    except Exception as e:
        print(f"Error providing feedback: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python provide_feedback_from_file.py <session_id> <feedback_file> [hypothesis_id1 hypothesis_id2 ...]")
        sys.exit(1)
    
    session_id = sys.argv[1]
    feedback_file = sys.argv[2]
    hypothesis_ids = sys.argv[3:] if len(sys.argv) > 3 else None
    
    asyncio.run(provide_feedback_from_file(session_id, feedback_file, hypothesis_ids)) 