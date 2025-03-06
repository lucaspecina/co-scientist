#!/usr/bin/env python
"""
Simple script to stop all active sessions by directly modifying the session files.
This avoids the complex error handling of the main management tool.
"""

import os
import json
import glob
from datetime import datetime

def stop_all_sessions():
    """Stop all active sessions by changing their state in the session files."""
    print("Stopping all active sessions directly...")
    
    # Check both possible session storage locations
    session_files = []
    session_files.extend(glob.glob("data/sessions/*.json"))
    session_files.extend(glob.glob("data/session_*.json"))
    
    if not session_files:
        print("No session files found.")
        return
    
    stopped_count = 0
    already_stopped_count = 0
    error_count = 0
    
    for file_path in session_files:
        try:
            # Extract session ID for reporting
            filename = os.path.basename(file_path)
            if filename.startswith("session_"):
                session_id = filename[8:].replace(".json", "")
            else:
                session_id = filename.replace(".json", "")
                
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    session_data = json.load(f)
                    
                    # Check if it's an active session that needs stopping
                    current_state = None
                    for state_key in ["state", "status"]:
                        if state_key in session_data:
                            current_state = session_data[state_key]
                            state_field = state_key
                            break
                            
                    if current_state is None:
                        print(f"  - {session_id}: Unknown state structure (no state/status field)")
                        error_count += 1
                        continue
                        
                    # Only modify active sessions
                    active_states = ["active", "evolving", "running", "initial"]
                    if current_state.lower() in active_states:
                        # Update the state to stopped
                        session_data[state_field] = "stopped"
                        
                        # Add stopped_at timestamp if not present
                        if "stopped_at" not in session_data:
                            session_data["stopped_at"] = datetime.now().isoformat()
                            
                        # Write the updated session data back to the file
                        with open(file_path, 'w', encoding='utf-8') as f_write:
                            json.dump(session_data, f_write, indent=2)
                            
                        print(f"  - {session_id}: Successfully stopped")
                        stopped_count += 1
                    else:
                        print(f"  - {session_id}: Already inactive (state: {current_state})")
                        already_stopped_count += 1
                        
                except json.JSONDecodeError:
                    print(f"  - {session_id}: Invalid JSON format (could not stop)")
                    error_count += 1
                    
        except Exception as e:
            print(f"  - Error processing {os.path.basename(file_path)}: {str(e)}")
            error_count += 1
    
    print("\nSummary:")
    print(f"  - Sessions stopped: {stopped_count}")
    print(f"  - Sessions already inactive: {already_stopped_count}")
    print(f"  - Sessions with errors: {error_count}")
    print(f"  - Total sessions processed: {len(session_files)}")

if __name__ == "__main__":
    stop_all_sessions() 