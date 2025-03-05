#!/usr/bin/env python
"""
AI Co-Scientist Monitoring Tool

This script provides a simple monitoring interface for AI Co-Scientist sessions,
allowing users to track progress and activities in real-time.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from core.memory.memory_manager import MemoryManager
from core.memory.file_system_memory_manager import FileSystemMemoryManager
from core.memory.mongodb_memory_manager import MongoDBMemoryManager
from core.workflow.supervisor import WorkflowState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitor.log')
    ]
)

logger = logging.getLogger("co_scientist_monitor")

# ANSI color codes for terminal output
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
}

class SessionMonitor:
    """Monitor for AI Co-Scientist sessions."""
    
    def __init__(self, session_id: str, mode: str = "terminal"):
        """
        Initialize the session monitor.
        
        Args:
            session_id: ID of the session to monitor
            mode: Monitoring mode ("terminal" or "web")
        """
        self.session_id = session_id
        self.mode = mode
        self.memory_manager = None
        self.last_state = None
        self.last_update_time = None
        self.token_usage = 0
        
    async def startup(self):
        """Initialize the monitor."""
        # Initialize memory manager
        try:
            # Try MongoDB first
            self.memory_manager = MongoDBMemoryManager()
            await self.memory_manager.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize MongoDB memory manager: {str(e)}")
            logger.info("Falling back to file system memory manager")
            
            # Fall back to file system
            self.memory_manager = FileSystemMemoryManager()
            await self.memory_manager.initialize()
            
        logger.info(f"Monitoring session: {self.session_id}")
    
    async def run(self):
        """Run the monitor."""
        if self.mode == "terminal":
            await self.run_terminal_monitor()
        elif self.mode == "web":
            print(f"{COLORS['YELLOW']}Web monitoring mode is not implemented yet. Falling back to terminal mode.{COLORS['RESET']}")
            await self.run_terminal_monitor()
        else:
            print(f"{COLORS['RED']}Unknown monitoring mode: {self.mode}{COLORS['RESET']}")
            sys.exit(1)
    
    async def run_terminal_monitor(self):
        """Run the terminal-based monitor."""
        print(f"\n{COLORS['BOLD']}{COLORS['CYAN']}AI Co-Scientist Session Monitor{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}Session ID: {COLORS['YELLOW']}{self.session_id}{COLORS['RESET']}\n")
        
        try:
            while True:
                # Get session data
                session_data = await self.memory_manager.get_session(self.session_id)
                
                if not session_data:
                    print(f"{COLORS['RED']}Session not found: {self.session_id}{COLORS['RESET']}")
                    break
                
                # Display session information
                self.display_session_info(session_data)
                
                # Check if session is completed
                state = session_data.get("state", "unknown")
                if state in ["completed", "failed", "stopped"]:
                    print(f"\n{COLORS['BOLD']}{COLORS['GREEN']}Session {state}. Monitoring stopped.{COLORS['RESET']}")
                    break
                
                # Wait before next update
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n{COLORS['YELLOW']}Monitoring stopped by user.{COLORS['RESET']}")
        except Exception as e:
            print(f"\n{COLORS['RED']}Error during monitoring: {str(e)}{COLORS['RESET']}")
        finally:
            # Clean up
            if self.memory_manager:
                await self.memory_manager.shutdown()
    
    def display_session_info(self, session_data: Dict[str, Any]):
        """Display session information in the terminal."""
        # Clear screen (Windows/PowerShell compatible)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Get basic session info
        state = session_data.get("state", "unknown")
        goal = session_data.get("goal", "No goal specified")
        domain = session_data.get("domain", "No domain specified")
        created_at = session_data.get("created_at", "Unknown")
        iterations = session_data.get("iterations_completed", 0)
        max_iterations = session_data.get("max_iterations", 0)
        
        # Get current activity
        current_activity = session_data.get("current_activity", "Idle")
        
        # Get token usage if available
        new_token_usage = session_data.get("token_usage", 0)
        if new_token_usage > self.token_usage:
            self.token_usage = new_token_usage
        
        # Format timestamps
        if isinstance(created_at, (int, float)):
            created_at = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S")
        
        # Display header
        print(f"{COLORS['BOLD']}{COLORS['CYAN']}AI Co-Scientist Session Monitor{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}Session ID: {COLORS['YELLOW']}{self.session_id}{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}Created: {COLORS['RESET']}{created_at}")
        print(f"{COLORS['BOLD']}Domain: {COLORS['RESET']}{domain}")
        print(f"{COLORS['BOLD']}Goal: {COLORS['RESET']}{goal}")
        print()
        
        # Display state with color
        state_color = {
            "initializing": COLORS["YELLOW"],
            "running": COLORS["GREEN"],
            "paused": COLORS["YELLOW"],
            "completed": COLORS["GREEN"],
            "failed": COLORS["RED"],
            "stopped": COLORS["RED"],
        }.get(state, COLORS["RESET"])
        
        print(f"{COLORS['BOLD']}State: {state_color}{state.upper()}{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}Current Activity: {COLORS['CYAN']}{current_activity}{COLORS['RESET']}")
        print(f"{COLORS['BOLD']}Iterations: {COLORS['RESET']}{iterations}/{max_iterations}")
        print(f"{COLORS['BOLD']}Token Usage: {COLORS['RESET']}{self.token_usage} tokens")
        print()
        
        # Display hypotheses if available
        hypotheses = session_data.get("hypotheses", [])
        if hypotheses:
            print(f"{COLORS['BOLD']}{COLORS['MAGENTA']}Top Hypotheses:{COLORS['RESET']}")
            
            # Sort hypotheses by score (descending)
            sorted_hypotheses = sorted(
                hypotheses, 
                key=lambda h: h.get("score", 0), 
                reverse=True
            )
            
            # Display top 3 hypotheses
            for i, hypothesis in enumerate(sorted_hypotheses[:3], 1):
                title = hypothesis.get("title", "Untitled")
                score = hypothesis.get("score", 0)
                print(f"{i}. {COLORS['BOLD']}{title}{COLORS['RESET']} (Score: {score:.2f})")
            
            print()
        
        # Display recent logs if available
        logs = session_data.get("logs", [])
        if logs:
            print(f"{COLORS['BOLD']}{COLORS['BLUE']}Recent Activity:{COLORS['RESET']}")
            
            # Display last 5 logs
            for log in logs[-5:]:
                timestamp = log.get("timestamp", "")
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                
                message = log.get("message", "")
                level = log.get("level", "info")
                
                level_color = {
                    "debug": COLORS["RESET"],
                    "info": COLORS["CYAN"],
                    "warning": COLORS["YELLOW"],
                    "error": COLORS["RED"],
                }.get(level.lower(), COLORS["RESET"])
                
                print(f"{timestamp} {level_color}{message}{COLORS['RESET']}")
            
            print()
        
        # Display feedback if available
        feedback = session_data.get("feedback_history", [])
        if feedback:
            print(f"{COLORS['BOLD']}{COLORS['GREEN']}Feedback History:{COLORS['RESET']}")
            
            # Display last feedback
            last_feedback = feedback[-1]
            feedback_text = last_feedback.get("text", "")
            feedback_time = last_feedback.get("timestamp", "")
            
            if isinstance(feedback_time, (int, float)):
                feedback_time = datetime.fromtimestamp(feedback_time).strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"{feedback_time}: {feedback_text[:100]}{'...' if len(feedback_text) > 100 else ''}")
            
            print()
        
        # Update tracking variables
        self.last_state = state
        self.last_update_time = time.time()
        
        # Display footer
        print(f"{COLORS['YELLOW']}Press Ctrl+C to stop monitoring{COLORS['RESET']}")


async def main():
    """Main entry point for the monitor."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="AI Co-Scientist Session Monitor"
    )
    parser.add_argument("--session", type=str, required=True, help="Session ID to monitor")
    parser.add_argument("--mode", type=str, default="terminal", choices=["terminal", "web"], 
                        help="Monitoring mode (terminal or web)")
    
    args = parser.parse_args()
    
    # Create and run monitor
    monitor = SessionMonitor(args.session, args.mode)
    await monitor.startup()
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main()) 