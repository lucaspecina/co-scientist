#!/usr/bin/env python
"""
Simple script to delete all session files.
This avoids the complex error handling of the main management tool.
"""

import os
import glob
import sys

def delete_all_sessions(force=False):
    """Delete all sessions from the file system."""
    print("Preparing to delete all sessions...")
    
    # Check both possible session storage locations
    session_files = []
    session_files.extend(glob.glob("data/sessions/*.json"))
    session_files.extend(glob.glob("data/session_*.json"))
    
    if not session_files:
        print("No session files found.")
        return
    
    print(f"Found {len(session_files)} session files.")
    
    # Confirm deletion unless force is True
    if not force:
        confirm = input(f"Are you sure you want to delete all {len(session_files)} sessions? (y/N): ")
        if confirm.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return
    
    # Delete the files
    deleted_count = 0
    error_count = 0
    
    for file_path in session_files:
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"  - Deleted: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  - Error deleting {os.path.basename(file_path)}: {str(e)}")
            error_count += 1
    
    print("\nSummary:")
    print(f"  - Sessions deleted: {deleted_count}")
    print(f"  - Errors encountered: {error_count}")
    print(f"  - Total sessions processed: {len(session_files)}")

if __name__ == "__main__":
    # Check for --force argument
    force = "--force" in sys.argv
    delete_all_sessions(force) 