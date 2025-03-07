#!/usr/bin/env python
"""
Session File Examiner for AI Co-Scientist

This script examines a specific session file in detail to help diagnose any issues.
"""

import os
import json
import sys
import binascii

def examine_session(session_id):
    """
    Examine a specific session file in detail.
    
    Args:
        session_id: The ID of the session to examine
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
        print(f"Error: Session file for '{session_id}' not found.")
        print("Checked these locations:")
        for path in potential_paths:
            print(f"  - {path}")
        return
    
    print(f"Examining session file: {file_path}")
    print("-" * 80)
    
    # Get file info
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    
    try:
        # Try to read the file as text
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Content length: {len(content)} characters")
        
        if not content.strip():
            print("Warning: File is empty or contains only whitespace")
            return
        
        # Try to parse as JSON
        try:
            session_data = json.loads(content)
            print("Successfully parsed as JSON")
            
            # Print key information
            print("\nSession Details:")
            print("-" * 80)
            
            # Print keys and their types
            print("Keys in the session file:")
            for key, value in session_data.items():
                value_type = type(value).__name__
                value_preview = str(value)
                if len(value_preview) > 50:
                    value_preview = value_preview[:47] + "..."
                print(f"  - {key} ({value_type}): {value_preview}")
            
            # Print status/state if available
            status = None
            for key in ["state", "status"]:
                if key in session_data and session_data[key]:
                    status = session_data[key]
                    break
            
            if status:
                print(f"\nStatus/State: {status}")
            else:
                print("\nStatus/State: Not specified")
                
            # Print goal if available
            goal = None
            for key in ["goal", "research_goal", "research_question"]:
                if key in session_data and session_data[key]:
                    goal = session_data[key]
                    break
            
            if goal:
                print(f"Goal: {goal}")
            else:
                print("Goal: Not specified")
                
            # Sample hypothesis structure
            if "hypotheses" in session_data and session_data["hypotheses"]:
                print("\nSample hypothesis structure:")
                print("-" * 80)
                hyp = session_data["hypotheses"][0]
                for key, value in hyp.items():
                    value_type = type(value).__name__
                    value_preview = str(value)
                    if len(value_preview) > 100:
                        value_preview = value_preview[:97] + "..."
                    print(f"  - {key} ({value_type}): {value_preview}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            print("\nFirst 200 characters of the file:")
            print(content[:200])
            
            if len(content) > 200:
                print("...")
            
    except UnicodeDecodeError:
        print("File contains non-UTF-8 characters. Attempting to read as binary...")
        
        try:
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            
            print(f"Binary content length: {len(binary_data)} bytes")
            print("\nFirst 100 bytes as hex:")
            hex_data = binascii.hexlify(binary_data[:100]).decode('ascii')
            for i in range(0, len(hex_data), 2):
                if i % 32 == 0:
                    print("")
                print(f"{hex_data[i:i+2]}", end=" ")
            print("\n...")
            
        except Exception as e:
            print(f"Error reading binary data: {str(e)}")
    
    except Exception as e:
        print(f"Error examining file: {str(e)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python examine_session.py <session_id>")
        print("\nExample:")
        print("  python examine_session.py 1741182035")
        print("  python examine_session.py session_1741182035")
        return
    
    session_id = sys.argv[1]
    examine_session(session_id)

if __name__ == "__main__":
    main() 