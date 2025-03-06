#!/usr/bin/env python
"""
Quick Session Lister for AI Co-Scientist

A very simple script to list all sessions with minimal dependencies.
This avoids all the complex error handling and processing of the main tool.
"""

import os
import json
import glob
import sys
from typing import Dict, Any, List
from datetime import datetime

def list_sessions():
    """List all sessions directly from the file system."""
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
                    
                except json.JSONDecodeError as e:
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

if __name__ == "__main__":
    list_sessions() 