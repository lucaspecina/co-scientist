#!/usr/bin/env python
"""
Check Hypothesis Scores for AI Co-Scientist

This script displays the scores of hypotheses in a session file.
"""

import os
import json
import sys
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_hypothesis_scores(session_id: str) -> bool:
    """
    Display hypothesis scores from a session file.
    
    Args:
        session_id: The ID of the session to check
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Add 'session_' prefix if not already present
    if not session_id.startswith("session_"):
        session_id = f"session_{session_id}"
    
    # Check potential file locations
    potential_paths = [
        f"data/sessions/{session_id}.json",
        f"data/{session_id}.json"
    ]
    
    file_path = None
    for path in potential_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if not file_path:
        logger.error(f"Session file for '{session_id}' not found.")
        logger.error("Checked these locations:")
        for path in potential_paths:
            logger.error(f"  - {path}")
        return False
    
    logger.info(f"Checking hypothesis scores in session file: {file_path}")
    
    try:
        # Read the session file
        with open(file_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Display top hypotheses
        if "top_hypotheses" in session_data:
            print("\nTop hypotheses:")
            print("-" * 80)
            top_ids = session_data.get("top_hypotheses", [])
            print(f"Top hypothesis IDs: {top_ids}")
        
        # Display all hypotheses and their scores
        if "hypotheses" in session_data:
            hypotheses = session_data["hypotheses"]
            print(f"\nFound {len(hypotheses)} hypotheses in the session file")
            print("-" * 80)
            
            # Sort hypotheses by score
            sorted_hypotheses = sorted(hypotheses, key=lambda h: h.get("score", 0), reverse=True)
            
            # Display scores for each hypothesis
            for i, hypothesis in enumerate(sorted_hypotheses[:10]):  # Show top 10
                hyp_id = hypothesis.get("id", "unknown")
                score = hypothesis.get("score", 0)
                confidence = hypothesis.get("confidence", "N/A")
                status = hypothesis.get("status", "N/A")
                text = hypothesis.get("text", "")
                
                # Truncate long text
                if len(text) > 100:
                    text = text[:97] + "..."
                
                print(f"\n{i+1}. Score: {score}")
                print(f"   Text: {text}")
                print(f"   Confidence: {confidence}")
                print(f"   Status: {status}")
                print(f"   ID: {hyp_id}")
                
                # Display criteria scores if available
                if "scores" in hypothesis and hypothesis["scores"]:
                    print("   Criteria scores:")
                    for criterion, score in hypothesis["scores"].items():
                        print(f"     - {criterion}: {score}")
            
            return True
        else:
            logger.error("No hypotheses found in the session file")
            return False
        
    except Exception as e:
        logger.error(f"Error checking hypothesis scores: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_hypothesis_scores.py <session_id>")
        print("\nExample:")
        print("  python check_hypothesis_scores.py 1741309717")
        print("  python check_hypothesis_scores.py session_1741309717")
        return
    
    session_id = sys.argv[1]
    success = check_hypothesis_scores(session_id)
    
    if success:
        print(f"\nSuccessfully checked scores for session {session_id}")
    else:
        print(f"\nFailed to check scores for session {session_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()