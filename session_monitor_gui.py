#!/usr/bin/env python
"""
AI Co-Scientist GUI Monitor

This script provides a graphical user interface for monitoring AI Co-Scientist sessions
in real-time, while allowing terminal commands to be used independently.
"""

import asyncio
import json
import logging
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading
import re
import glob
import traceback

# Define version
VERSION = "1.0.0"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gui_monitor.log')
    ]
)
logger = logging.getLogger("gui_monitor")

# Import core components
try:
    from core.memory.file_system_memory_manager import FileSystemMemoryManager
    from core.controller import CoScientistController
    from core.workflow.supervisor import WorkflowState
    CORE_IMPORTS_SUCCESS = True
except ImportError as e:
    logger.error(f"Failed to import core components: {e}")
    CORE_IMPORTS_SUCCESS = False
    # Create dummy classes for the imports that failed
    class DummyFileSystemMemoryManager:
        """Dummy class when FileSystemMemoryManager is not available."""
        async def list_sessions(self):
            logger.warning("Using dummy memory manager - list_sessions")
            return []
            
        async def get_session(self, session_id):
            logger.warning("Using dummy memory manager - get_session")
            return None
            
        async def update_session(self, session_id, data):
            logger.warning("Using dummy memory manager - update_session")
            return False
            
        async def delete_session(self, session_id):
            logger.warning("Using dummy memory manager - delete_session")
            return False
    
    # Use the dummy classes
    FileSystemMemoryManager = DummyFileSystemMemoryManager
    
    # We'll show a warning but not exit
    def show_import_warning():
        messagebox.showwarning(
            "Import Warning", 
            "Failed to import core components. Running in direct filesystem mode.\n"
            "Limited functionality will be available."
        )
        # Schedule this to run after the GUI is initialized to avoid errors
        # tk.Tk().after(1000, lambda: messagebox.showwarning(...)) won't work here
        # so we'll set a flag to show the warning later

class SessionMonitorGUI:
    """GUI application for monitoring AI Co-Scientist sessions in real-time."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AI Co-Scientist Session Monitor")
        self.root.geometry("900x600")
        self.root.minsize(800, 500)
        
        # Initialize memory manager and controller
        self.memory_manager = None
        self.controller = None
        
        # Set up refresh rate (in milliseconds)
        self.refresh_rate = 5000  # 5 seconds
        self.auto_refresh = True
        
        # Track if the application is closing
        self.is_closing = False
        
        # Debug mode
        self.debug_mode = True
        
        # Flag for using direct filesystem access
        self.use_direct_fs = not CORE_IMPORTS_SUCCESS  # Default to direct FS if imports failed
        
        # Create the GUI layout
        self.create_widgets()
        
        # Show import warning if needed
        if not CORE_IMPORTS_SUCCESS:
            self.root.after(1000, lambda: messagebox.showwarning(
                "Import Warning", 
                "Failed to import core components. Running in direct filesystem mode.\n"
                "Limited functionality will be available."
            ))
        
        # Start the initialization in a separate thread
        self.status_label.config(text="Initializing session monitor...")
        threading.Thread(target=self.initialize_async, daemon=True).start()
    
    def initialize_async(self):
        """Initialize the async components in a separate thread."""
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Check data directory first
            data_exists = self.check_data_directory()
            
            # If we're using direct filesystem, don't bother with the memory manager
            if self.use_direct_fs:
                self.root.after(0, lambda: self.status_label.config(text="Using direct filesystem access mode"))
                # Schedule the first refresh directly
                self.root.after(1000, self.refresh_data)
                return
                
            # Try full approach with memory manager
            try:
                # Initialize memory manager directly here instead of using controller
                if self.debug_mode:
                    self.root.after(0, lambda: self.status_label.config(text="Creating memory manager..."))
                
                self.memory_manager = FileSystemMemoryManager()
                
                if self.debug_mode:
                    self.root.after(0, lambda: self.status_label.config(text="Memory manager initialized. Testing connection..."))
                
                # Test the memory manager by listing sessions
                session_ids = loop.run_until_complete(self.memory_manager.list_sessions())
                
                if self.debug_mode:
                    self.root.after(0, lambda: self.status_label.config(text=f"Found {len(session_ids)} sessions. Starting refresh..."))
            except Exception as e:
                logger.error(f"Memory manager initialization failed: {str(e)}", exc_info=True)
                self.use_direct_fs = True
                self.root.after(0, lambda: self.status_label.config(text=f"Using direct filesystem approach instead"))
            
            # Schedule the first refresh directly on the main thread
            self.root.after(1000, self.refresh_data)
        except Exception as e:
            error_msg = f"Initialization error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
            
            # Always fall back to direct file system access on errors
            self.use_direct_fs = True
            self.root.after(3000, self.refresh_data)
    
    def check_data_directory(self):
        """Check the data directory to verify we can read session files."""
        data_dir = os.path.join(os.getcwd(), "data")
        sessions_dir = os.path.join(data_dir, "sessions")
        
        # Check if directories exist
        if not os.path.exists(data_dir):
            error_msg = f"Data directory not found at {data_dir}. Please run this from the project root."
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
            logger.error(error_msg)
            return False
            
        if not os.path.exists(sessions_dir):
            error_msg = f"Sessions directory not found at {sessions_dir}. No sessions may have been created yet."
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
            logger.warning(error_msg)
            return False
        
        # Try to list files in sessions directory
        try:
            session_files = os.listdir(sessions_dir)
            if not session_files:
                warn_msg = "No session files found. Please create a session first."
                self.root.after(0, lambda: self.status_label.config(text=warn_msg))
                logger.warning(warn_msg)
                return False
                
            # Try to read one session file to verify permissions
            if session_files:
                sample_file = os.path.join(sessions_dir, session_files[0])
                with open(sample_file, 'r') as f:
                    sample_data = f.read()
                    if self.debug_mode:
                        logger.info(f"Successfully read sample session file: {sample_file}")
                        logger.info(f"Sample data preview: {sample_data[:100]}")
                        
            # Log success message
            logger.info(f"Found {len(session_files)} session files in {sessions_dir}")
            return True
        except Exception as e:
            error_msg = f"Error reading session files: {str(e)}"
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
            logger.error(error_msg, exc_info=True)
            return False

    def create_widgets(self):
        """Create the GUI widgets."""
        # Create a main frame for the entire application
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a frame for the top controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Refresh button
        refresh_btn = ttk.Button(controls_frame, text="Refresh", command=self.manual_refresh)
        refresh_btn.pack(side="left", padx=5)
        
        # Auto-refresh checkbox
        self.auto_refresh_var = tk.BooleanVar(value=True)
        auto_refresh_check = ttk.Checkbutton(
            controls_frame, 
            text="Auto-refresh", 
            variable=self.auto_refresh_var,
            command=self.toggle_auto_refresh
        )
        auto_refresh_check.pack(side="left", padx=5)
        
        # Filter options
        ttk.Label(controls_frame, text="Filter:").pack(side="left", padx=5)
        
        self.filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.filter_var,
            values=["all", "active", "completed", "failed", "stopped"],
            width=10
        )
        filter_combo.pack(side="left", padx=5)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self.manual_refresh())
        
        # Debug mode checkbox
        self.debug_var = tk.BooleanVar(value=self.debug_mode)
        debug_check = ttk.Checkbutton(
            controls_frame, 
            text="Debug Mode", 
            variable=self.debug_var,
            command=self.toggle_debug_mode
        )
        debug_check.pack(side="right", padx=5)
        
        # Create notebook (tabbed interface)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create Sessions tab
        sessions_frame = ttk.Frame(notebook)
        notebook.add(sessions_frame, text="Sessions")
        
        # Create a frame for the sessions table
        table_frame = ttk.Frame(sessions_frame)
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for sessions with scrollbar
        columns = ("ID", "State", "Goal", "Created", "Tokens")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        # Configure column headings
        self.tree.heading("ID", text="Session ID")
        self.tree.heading("State", text="State")
        self.tree.heading("Goal", text="Goal")
        self.tree.heading("Created", text="Created")
        self.tree.heading("Tokens", text="Tokens")
        
        # Configure column widths
        self.tree.column("ID", width=150, minwidth=100)
        self.tree.column("State", width=80, minwidth=80)
        self.tree.column("Goal", width=350, minwidth=200)
        self.tree.column("Created", width=150, minwidth=150)
        self.tree.column("Tokens", width=80, minwidth=80)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout for treeview and scrollbars
        self.tree.grid(column=0, row=0, sticky="nsew")
        vsb.grid(column=1, row=0, sticky="ns")
        hsb.grid(column=0, row=1, sticky="ew")
        
        # Configure grid weights
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Configure tags for different states
        self.tree.tag_configure("active", background="#e6ffe6")  # Light green
        self.tree.tag_configure("completed", background="#e6e6ff")  # Light blue
        self.tree.tag_configure("failed", background="#ffe6e6")  # Light red
        self.tree.tag_configure("stopped", background="#fff6e6")  # Light orange
        self.tree.tag_configure("error", background="#ffcccc")  # Bright red for errors
        self.tree.tag_configure("unknown", background="#f0f0f0")  # Gray for unknown
        
        # Bind select event
        self.tree.bind("<<TreeviewSelect>>", self.on_session_select)
        
        # Create Details tab
        details_frame = ttk.Frame(notebook)
        notebook.add(details_frame, text="Session Details")
        
        # Create details display
        self.details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, height=15)
        self.details_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create Log tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Logs")
        
        # Create log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(log_controls, text="Session:").pack(side="left", padx=5)
        
        self.log_session_var = tk.StringVar()
        self.log_session_combo = ttk.Combobox(log_controls, textvariable=self.log_session_var, width=50)
        self.log_session_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        view_logs_btn = ttk.Button(log_controls, text="View Logs", command=self.view_selected_logs)
        view_logs_btn.pack(side="left", padx=5)
        
        # Auto-refresh logs checkbox
        self.auto_refresh_logs_var = tk.BooleanVar(value=False)
        auto_refresh_logs_check = ttk.Checkbutton(
            log_controls, 
            text="Auto-refresh", 
            variable=self.auto_refresh_logs_var
        )
        auto_refresh_logs_check.pack(side="left", padx=5)
        
        # Create log display
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure text tags for different message types
        self.log_text.tag_configure("SYSTEM", foreground="blue")
        self.log_text.tag_configure("INFO", foreground="green")
        self.log_text.tag_configure("ERROR", foreground="red")
        self.log_text.tag_configure("WARNING", foreground="orange")
        self.log_text.tag_configure("OUTPUT", foreground="black")
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.pack(fill="x", padx=5, pady=5)
        
        # Initialize dictionary to store session data
        self.session_data = {}
        self.session_display_to_id = {}
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh functionality."""
        self.auto_refresh = self.auto_refresh_var.get()
        if self.auto_refresh:
            # Start auto-refresh
            self.status_label.config(text="Auto-refresh enabled")
            self.root.after(self.refresh_rate, self.refresh_data)
        else:
            # Stop auto-refresh
            self.status_label.config(text="Auto-refresh disabled")
            if hasattr(self, 'after_handle'):
                self.root.after_cancel(self.after_handle)

    def toggle_debug_mode(self):
        """Toggle debug mode."""
        self.debug_mode = self.debug_var.get()
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.info("Debug mode enabled")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug mode disabled")

    def manual_refresh(self):
        """Manually refresh the data."""
        self.status_label.config(text="Refreshing data...")
        # Cancel any pending auto-refresh
        if hasattr(self, 'after_handle'):
            self.root.after_cancel(self.after_handle)
        # Start a manual refresh in the background
        threading.Thread(target=lambda: self.fetch_and_update_data(manual=True), daemon=True).start()

    def refresh_data(self):
        """Auto-refresh data."""
        if self.auto_refresh:
            self.status_label.config(text="Auto-refreshing data...")
            # Start a refresh in the background
            threading.Thread(target=self.fetch_and_update_data, daemon=True).start()

    def on_session_select(self, event):
        """Handle selection of a session in the treeview."""
        try:
            selected_items = self.tree.selection()
            if not selected_items:
                return
                
            # Get the selected session ID
            session_id = selected_items[0]
            
            # Get the session data
            if session_id in self.session_data:
                session = self.session_data[session_id]
                
                # Display session details
                self.display_session_details(session)
        except Exception as e:
            logger.error(f"Error handling session selection: {str(e)}", exc_info=True)

    def display_session_details(self, session):
        """Display details of the selected session."""
        try:
            # Clear previous content
            self.details_text.configure(state=tk.NORMAL)
            self.details_text.delete(1.0, tk.END)
            
            # Format session data
            details = []
            details.append(f"Session ID: {session.get('session_id', 'Unknown')}")
            details.append(f"State: {session.get('state', 'Unknown')}")
            details.append(f"Goal: {session.get('goal', 'No goal specified')}")
            
            # Format creation time
            created_at = session.get('created_at', 0)
            if isinstance(created_at, (int, float)):
                created_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))
                details.append(f"Created: {created_str}")
            
            # Add token usage if available
            tokens = session.get('tokens', {})
            if tokens:
                details.append("\nToken Usage:")
                for token_type, count in tokens.items():
                    details.append(f"  {token_type}: {count:,}")
                
                total_tokens = sum(tokens.values())
                details.append(f"\nTotal Tokens: {total_tokens:,}")
            
            # Add hypotheses if available
            hypotheses = session.get('hypotheses', [])
            if hypotheses:
                details.append("\nHypotheses:")
                for i, hypothesis in enumerate(hypotheses):
                    details.append(f"  {i+1}. {hypothesis.get('text', 'No text')}")
                    if 'confidence' in hypothesis:
                        details.append(f"     Confidence: {hypothesis.get('confidence', 0)}")
                    if 'tags' in hypothesis and hypothesis['tags']:
                        details.append(f"     Tags: {', '.join(hypothesis['tags'])}")
            
            # Add steps if available
            steps = session.get('steps', [])
            if steps:
                details.append("\nWorkflow Steps:")
                for i, step in enumerate(steps):
                    step_type = step.get('type', 'Unknown')
                    step_state = step.get('state', 'Unknown')
                    details.append(f"  {i+1}. [{step_state}] {step_type}")
                    if 'description' in step:
                        details.append(f"     {step.get('description', '')}")
            
            # Add any other important fields
            additional_fields = [
                ('completion_time', 'Completed'),
                ('error', 'Error'),
                ('source', 'Source'),
                ('user', 'User'),
                ('assigned_agent', 'Agent'),
            ]
            
            for field, label in additional_fields:
                if field in session and session[field]:
                    details.append(f"\n{label}: {session[field]}")
            
            # Display the formatted details
            self.details_text.insert(tk.END, "\n".join(details))
            self.details_text.configure(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Error displaying session details: {str(e)}", exc_info=True)
            self.details_text.configure(state=tk.NORMAL)
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, f"Error displaying session details: {str(e)}")
            self.details_text.configure(state=tk.DISABLED)

    def view_selected_logs(self):
        """View logs for the selected session."""
        try:
            # Get selected session from combobox
            selected = self.log_session_var.get()
            if not selected:
                return
                
            # Get the session ID from the selection
            session_id = None
            if selected in self.session_display_to_id:
                session_id = self.session_display_to_id[selected]
            else:
                # If not in the mapping, try to use it directly
                session_id = selected
                
            if not session_id:
                logger.error("No session ID found for selected item")
                return
                
            # Clear log text
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, f"Loading logs for session {session_id}...\n", "SYSTEM")
            
            # Start loading in a background thread
            threading.Thread(target=lambda: self.load_session_logs(session_id), daemon=True).start()
        except Exception as e:
            logger.error(f"Error viewing selected logs: {str(e)}", exc_info=True)
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"Error: {str(e)}\n", "ERROR")
            self.log_text.config(state=tk.DISABLED)

    def load_session_logs(self, session_id):
        """Load and display logs for a specific session."""
        try:
            # Try to find logs from various sources
            logs = []
            
            # First try to scan for log files
            logs = self.scan_for_logs(session_id)
            
            # Update the UI
            self.root.after(0, lambda: self.display_logs(logs))
        except Exception as e:
            logger.error(f"Error loading session logs: {str(e)}", exc_info=True)
            self.root.after(0, lambda: self.log_text.insert(tk.END, f"Error loading logs: {str(e)}\n", "ERROR"))

    def scan_for_logs(self, session_id):
        """Scan for log files related to a session."""
        logs = []
        try:
            # Determine possible locations for logs
            possible_locations = [
                os.path.expanduser(f"~/.co-scientist/sessions/{session_id}/logs"),
                os.path.join(os.getcwd(), ".co-scientist", "sessions", session_id, "logs"),
                os.path.join(os.path.dirname(os.getcwd()), ".co-scientist", "sessions", session_id, "logs"),
                # Add more potential locations
            ]
            
            for location in possible_locations:
                if os.path.exists(location) and os.path.isdir(location):
                    if self.debug_mode:
                        logger.info(f"Checking for logs in {location}")
                    
                    # Look for log files
                    for filename in sorted(os.listdir(location)):
                        if filename.endswith('.log') or filename.endswith('.txt'):
                            log_file = os.path.join(location, filename)
                            try:
                                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                                    content = f.read()
                                    logs.append({
                                        'file': filename,
                                        'content': content,
                                        'path': log_file
                                    })
                                    if self.debug_mode:
                                        logger.info(f"Found log file: {log_file}")
                            except Exception as e:
                                logger.error(f"Error reading log file {log_file}: {str(e)}", exc_info=True)
            
            # If no logs found, try the session directory directly
            if not logs:
                parent_locations = [
                    os.path.expanduser(f"~/.co-scientist/sessions/{session_id}"),
                    os.path.join(os.getcwd(), ".co-scientist", "sessions", session_id),
                    os.path.join(os.path.dirname(os.getcwd()), ".co-scientist", "sessions", session_id),
                ]
                
                for location in parent_locations:
                    if os.path.exists(location) and os.path.isdir(location):
                        if self.debug_mode:
                            logger.info(f"Checking for logs in parent directory {location}")
                        
                        # Look for log files
                        for filename in sorted(os.listdir(location)):
                            if filename.endswith('.log') or filename.endswith('.txt'):
                                log_file = os.path.join(location, filename)
                                try:
                                    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                                        content = f.read()
                                        logs.append({
                                            'file': filename,
                                            'content': content,
                                            'path': log_file
                                        })
                                        if self.debug_mode:
                                            logger.info(f"Found log file: {log_file}")
                                except Exception as e:
                                    logger.error(f"Error reading log file {log_file}: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error scanning for logs: {str(e)}", exc_info=True)
        
        return logs

    def display_logs(self, logs):
        """Display the loaded logs in the UI."""
        try:
            # Clear existing content
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            
            if not logs:
                self.log_text.insert(tk.END, "No logs found for this session.", "SYSTEM")
                self.log_text.config(state=tk.DISABLED)
                return
            
            # Display each log file
            for log in logs:
                filename = log.get('file', 'Unknown')
                content = log.get('content', '')
                
                # Add header for this log file
                self.log_text.insert(tk.END, f"\n=== {filename} ===\n\n", "SYSTEM")
                
                # Add the content with appropriate tags
                lines = content.split('\n')
                for line in lines:
                    line_tag = "OUTPUT"  # Default tag
                    
                    # Try to detect the log level from the line
                    lower_line = line.lower()
                    if "error" in lower_line or "exception" in lower_line:
                        line_tag = "ERROR"
                    elif "warn" in lower_line:
                        line_tag = "WARNING"
                    elif "info" in lower_line:
                        line_tag = "INFO"
                    elif "system" in lower_line or "starting" in lower_line or "initialized" in lower_line:
                        line_tag = "SYSTEM"
                    
                    self.log_text.insert(tk.END, line + "\n", line_tag)
            
            # Scroll to the beginning
            self.log_text.see("1.0")
            self.log_text.config(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Error displaying logs: {str(e)}", exc_info=True)
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"Error displaying logs: {str(e)}\n", "ERROR")
            self.log_text.config(state=tk.DISABLED)

    def scan_for_additional_logs(self, session_id):
        """Scan for additional log files in the workspace."""
        additional_logs = []
        try:
            # Search in common log locations
            log_locations = [
                "logs",
                "log",
                "data/logs",
                ".logs",
                "output",
                "results"
            ]
            
            for location in log_locations:
                if os.path.exists(location) and os.path.isdir(location):
                    for filename in os.listdir(location):
                        if session_id in filename and (filename.endswith('.log') or filename.endswith('.txt')):
                            log_file = os.path.join(location, filename)
                            try:
                                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                                    content = f.read()
                                    additional_logs.append({
                                        'file': filename,
                                        'content': content,
                                        'path': log_file
                                    })
                            except Exception as e:
                                logger.error(f"Error reading additional log file {log_file}: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error scanning for additional logs: {str(e)}", exc_info=True)
        
        return additional_logs

    def fetch_and_update_data(self, manual=False):
        """Fetch data in a separate thread and update the UI."""
        try:
            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Get filter value from UI thread-safely
            filter_value = self.filter_var.get()
            filter_state = None if filter_value == "all" else filter_value
            
            # Status update
            status_msg = f"Fetching sessions (filter: {filter_value})..."
            if self.debug_mode:
                logger.info(status_msg)
            self.root.after(0, lambda: self.status_label.config(text=status_msg))
            
            # Variable to store the sessions
            sessions = []
            
            # Use the appropriate method to get sessions
            if self.use_direct_fs:
                # Use direct filesystem access
                if self.debug_mode:
                    logger.info(f"Using direct filesystem access to find sessions")
                
                # First try the normal filesystem method
                sessions = self.list_sessions_from_filesystem(filter_state)
                
                # If that doesn't work, try the more aggressive scan
                if not sessions:
                    if self.debug_mode:
                        logger.info(f"No sessions found with standard method, trying aggressive scan")
                    sessions = self.find_any_sessions_anywhere()
                
                if self.debug_mode:
                    logger.info(f"Direct filesystem found {len(sessions)} sessions")
            else:
                # Try using the regular memory manager
                try:
                    # List all sessions first
                    session_ids = loop.run_until_complete(self.memory_manager.list_sessions())
                    
                    if self.debug_mode:
                        self.root.after(0, lambda: self.status_label.config(text=f"Found {len(session_ids)} sessions, retrieving details..."))
                    
                    # Then get full data for each session
                    sessions = []
                    for session_id in session_ids:
                        try:
                            session_data = loop.run_until_complete(self.memory_manager.get_session(session_id))
                            
                            # Normalize the data
                            normalized_data = self.normalize_session_data(session_data)
                            
                            # Debug session data
                            if self.debug_mode:
                                logger.info(f"Session data for {session_id}: {str(normalized_data)[:200]}...")
                            
                            # Filter by state if specified
                            if filter_state and normalized_data.get("state") != filter_state:
                                continue
                                
                            sessions.append(normalized_data)
                        except Exception as e:
                            logger.error(f"Error getting data for session {session_id}: {str(e)}", exc_info=True)
                            
                            # Try direct file access as a fallback
                            try:
                                session_data = self.read_session_file_directly(session_id)
                                if session_data:
                                    # Filter by state if specified
                                    if filter_state and session_data.get("state") != filter_state:
                                        continue
                                        
                                    sessions.append(session_data)
                                else:
                                    # Add partial data so we at least see the session ID
                                    sessions.append({
                                        "session_id": session_id,
                                        "goal": f"[Error loading data: {str(e)}]",
                                        "state": "error",
                                        "created_at": time.time(),
                                        "tokens": {}
                                    })
                            except Exception as e2:
                                logger.error(f"Error with direct file access for {session_id}: {str(e2)}", exc_info=True)
                                # Add partial data so we at least see the session ID
                                sessions.append({
                                    "session_id": session_id,
                                    "goal": f"[Error loading data: {str(e)} and direct file access: {str(e2)}]",
                                    "state": "error",
                                    "created_at": time.time(),
                                    "tokens": {}
                                })
                except Exception as e:
                    logger.error(f"Memory manager approach failed: {str(e)}", exc_info=True)
                    
                    # If memory manager fails, try direct file system approach
                    sessions = self.list_sessions_from_filesystem(filter_state)
                    
                    # Set the flag for future use
                    self.use_direct_fs = True
            
            # If we still don't have any sessions, do a thorough search
            if not sessions:
                logger.warning("No sessions found with primary methods, trying broad directory scan")
                sessions = self.find_any_sessions_anywhere()
                
                # If still no sessions, create a dummy session just to show something
                if not sessions:
                    logger.warning("No sessions found at all, creating a dummy session for UI purposes")
                    sessions = [{
                        "session_id": "no_sessions_found",
                        "goal": "No sessions found. Please create a session first or check your data directory structure.",
                        "state": "unknown",
                        "created_at": time.time(),
                        "tokens": {}
                    }]
            
            # Update UI in the main thread
            self.root.after(0, lambda: self.update_sessions_table(sessions, manual))
        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
    
    def find_any_sessions_anywhere(self):
        """More aggressive search for session files in various locations."""
        sessions = []
        try:
            # Common locations where sessions might be stored
            search_roots = [
                os.path.expanduser("~"),
                os.getcwd(),
                os.path.dirname(os.getcwd()),
                # Add more locations if needed
            ]
            
            # Patterns to look for
            session_patterns = [
                "**/.co-scientist/sessions/*/session.json",
                "**/sessions/*/session.json",
                "**/co-scientist/**/sessions/*/session.json",
                "**/data/sessions/*.json",
            ]
            
            for root in search_roots:
                if self.debug_mode:
                    logger.info(f"Searching for sessions in: {root}")
                
                for pattern in session_patterns:
                    # Use glob to find all matching files
                    for session_file in glob.glob(os.path.join(root, pattern), recursive=True):
                        try:
                            # Extract session_id from the path
                            if "session.json" in session_file:
                                session_dir = os.path.dirname(session_file)
                                session_id = os.path.basename(session_dir)
                            else:
                                # For direct JSON files, use the filename without extension
                                session_id = os.path.basename(session_file).replace('.json', '')
                            
                            # Read the session file
                            with open(session_file, 'r', encoding='utf-8') as f:
                                session_data = json.load(f)
                                
                                # Normalize the data
                                normalized_data = self.normalize_session_data(session_data)
                                
                                # Add to our list if not already there
                                if not any(s.get("session_id") == normalized_data["session_id"] for s in sessions):
                                    sessions.append(normalized_data)
                                    
                                    if self.debug_mode:
                                        logger.info(f"Found session: {normalized_data['session_id']} at {session_file}")
                        except Exception as e:
                            logger.error(f"Error processing file {session_file}: {str(e)}", exc_info=True)
            
            if self.debug_mode:
                logger.info(f"Aggressive search found {len(sessions)} sessions")
        except Exception as e:
            logger.error(f"Error in aggressive session search: {str(e)}", exc_info=True)
        
        return sessions
    
    def list_sessions_from_filesystem(self, filter_state=None):
        """List all sessions by directly scanning the filesystem."""
        sessions = []
        try:
            # Try to find the session directory
            base_dirs = [
                os.path.expanduser("~/.co-scientist/sessions/"),
                os.path.join(os.getcwd(), ".co-scientist", "sessions"),
                os.path.join(os.path.dirname(os.getcwd()), ".co-scientist", "sessions"),
                os.path.join(os.getcwd(), "data", "sessions"),
                # Add more potential locations as needed
            ]
            
            for base_dir in base_dirs:
                if os.path.exists(base_dir) and os.path.isdir(base_dir):
                    if self.debug_mode:
                        logger.info(f"Scanning for sessions in: {base_dir}")
                    
                    # Check if this is a directory of session directories or a directory of session files
                    session_files_in_dir = [f for f in os.listdir(base_dir) if f.endswith('.json')]
                    
                    if session_files_in_dir:
                        # This directory contains session files directly
                        for filename in session_files_in_dir:
                            file_path = os.path.join(base_dir, filename)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    session_data = json.load(f)
                                    
                                    # Normalize the data
                                    normalized_data = self.normalize_session_data(session_data)
                                    
                                    # Filter by state if specified
                                    if filter_state and normalized_data.get("state") != filter_state:
                                        continue
                                    
                                    sessions.append(normalized_data)
                                    
                                    if self.debug_mode:
                                        logger.info(f"Found session from file: {file_path}")
                            except Exception as e:
                                logger.error(f"Error reading session file {file_path}: {str(e)}", exc_info=True)
                    else:
                        # Look for session directories
                        for session_id in os.listdir(base_dir):
                            session_dir = os.path.join(base_dir, session_id)
                            if os.path.isdir(session_dir):
                                try:
                                    # Try to read the session.json file
                                    session_data = self.read_session_file_directly(session_id, base_dir)
                                    if session_data:
                                        # Filter by state if specified
                                        if filter_state and session_data.get("state") != filter_state:
                                            continue
                                        
                                        sessions.append(session_data)
                                except Exception as e:
                                    logger.error(f"Error reading session {session_id}: {str(e)}", exc_info=True)
            
            if len(sessions) > 0:
                logger.info(f"Found {len(sessions)} session files in {os.getcwd()}")
            else:
                logger.warning(f"No sessions found in any standard location")
                
            if self.debug_mode:
                logger.info(f"Found {len(sessions)} sessions via filesystem")
        except Exception as e:
            logger.error(f"Error listing sessions from filesystem: {str(e)}", exc_info=True)
        
        return sessions
    
    def read_session_file_directly(self, session_id, base_dir=None):
        """Read a session file directly from the filesystem."""
        try:
            # Determine all possible locations
            possible_dirs = []
            if base_dir:
                possible_dirs.append(os.path.join(base_dir, session_id))
            else:
                possible_dirs.extend([
                    os.path.join(os.path.expanduser("~/.co-scientist/sessions/"), session_id),
                    os.path.join(os.getcwd(), ".co-scientist", "sessions", session_id),
                    os.path.join(os.path.dirname(os.getcwd()), ".co-scientist", "sessions", session_id),
                    os.path.join(os.getcwd(), "data", "sessions", session_id),
                    os.path.join(os.getcwd(), "data", "sessions"),
                    # Add more potential locations as needed
                ])
            
            # Check if session_id might be a full path to a JSON file
            if session_id.endswith('.json') and os.path.exists(session_id):
                with open(session_id, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    return self.normalize_session_data(session_data)
            
            # Try each location
            for session_dir in possible_dirs:
                if os.path.isdir(session_dir):
                    # Try to find a session.json file
                    session_file = os.path.join(session_dir, "session.json")
                    if os.path.exists(session_file):
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                            return self.normalize_session_data(session_data)
                
                # Also check if the directory contains JSON files directly
                elif os.path.isdir(os.path.dirname(session_dir)):
                    # Check if session_id is in the filename
                    parent_dir = os.path.dirname(session_dir)
                    for filename in os.listdir(parent_dir):
                        if session_id in filename and filename.endswith('.json'):
                            file_path = os.path.join(parent_dir, filename)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                session_data = json.load(f)
                                if self.debug_mode:
                                    logger.info(f"Successfully read sample session file: {file_path}")
                                    logger.info(f"Sample data preview: {json.dumps(session_data, indent=2)[:200]}...")
                                return self.normalize_session_data(session_data)
            
            # If not found in specific locations, try a broader search in data/sessions
            data_sessions_dir = os.path.join(os.getcwd(), "data", "sessions")
            if os.path.exists(data_sessions_dir) and os.path.isdir(data_sessions_dir):
                for filename in os.listdir(data_sessions_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(data_sessions_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                session_data = json.load(f)
                                # Check if this session matches the ID we're looking for
                                if session_data.get('id') == session_id or session_data.get('session_id') == session_id:
                                    if self.debug_mode:
                                        logger.info(f"Found session file matching ID: {file_path}")
                                    return self.normalize_session_data(session_data)
                        except Exception as e:
                            logger.error(f"Error reading potential session file {file_path}: {str(e)}", exc_info=True)
            
            return None
        except Exception as e:
            logger.error(f"Error reading session file for {session_id}: {str(e)}", exc_info=True)
            return None

    def normalize_session_data(self, session_data):
        """Normalize session data to handle different field names."""
        if not isinstance(session_data, dict):
            logger.error(f"Session data is not a dictionary: {type(session_data)}")
            return {
                "session_id": "Invalid",
                "goal": "Invalid session data format",
                "state": "error",
                "created_at": 0,
                "tokens": {}
            }
            
        # Create a deep copy to avoid modifying the original
        normalized = {}
        
        # First identify essential fields
        
        # Session ID
        if 'session_id' in session_data:
            normalized['session_id'] = session_data['session_id']
        elif 'id' in session_data:
            normalized['session_id'] = session_data['id']
        else:
            # Try to generate a session ID from timestamp
            normalized['session_id'] = f"session_{int(time.time())}"
            
        # Goal/Research Question
        if 'goal' in session_data and session_data['goal']:
            if isinstance(session_data['goal'], dict) and 'description' in session_data['goal']:
                normalized['goal'] = session_data['goal']['description']
            else:
                normalized['goal'] = str(session_data['goal'])
        elif 'research_goal' in session_data and session_data['research_goal']:
            normalized['goal'] = session_data['research_goal']
        elif 'prompt' in session_data and session_data['prompt']:
            normalized['goal'] = session_data['prompt']
        elif 'question' in session_data and session_data['question']:
            normalized['goal'] = session_data['question']
        elif 'description' in session_data and session_data['description']:
            normalized['goal'] = session_data['description']
        else:
            normalized['goal'] = f"[No goal specified for session {normalized['session_id']}]"
            
        # State/Status
        if 'state' in session_data:
            normalized['state'] = session_data['state']
        elif 'status' in session_data:
            normalized['state'] = session_data['status']
        elif 'workflow_state' in session_data:
            normalized['state'] = session_data['workflow_state']
        else:
            normalized['state'] = "unknown"
            
        # Creation Time
        if 'created_at' in session_data and session_data['created_at']:
            normalized['created_at'] = session_data['created_at']
        elif 'creation_time' in session_data and session_data['creation_time']:
            normalized['created_at'] = session_data['creation_time']
        elif 'timestamp' in session_data and session_data['timestamp']:
            normalized['created_at'] = session_data['timestamp']
        else:
            normalized['created_at'] = time.time()  # Default to current time
            
        # Token Usage
        if 'tokens' in session_data and session_data['tokens']:
            normalized['tokens'] = session_data['tokens']
        else:
            normalized['tokens'] = {}
            
        # Handle nested data
        for key in ['hypotheses', 'steps', 'outputs', 'errors']:
            if key in session_data:
                normalized[key] = session_data[key]
                
        # Log the normalized data for debugging
        if self.debug_mode:
            logger.info(f"Session data for {normalized['session_id']}: {json.dumps(normalized)[:200]}...")
            
        return normalized

    def update_sessions_table(self, sessions, manual=False):
        """Update the sessions table with new data."""
        try:
            # Clear the current data
            self.tree.delete(*self.tree.get_children())
            
            # Progress update
            status_msg = f"Displaying {len(sessions)} sessions..."
            if self.debug_mode:
                logger.info(status_msg)
            self.status_label.config(text=status_msg)
            
            # Process and display sessions
            for session in sorted(sessions, key=lambda x: x.get('created_at', 0), reverse=True):
                try:
                    # Extract session data
                    session_id = session.get("session_id", "unknown")
                    state = session.get("state", "unknown")
                    goal = session.get("goal", "No goal specified")
                    
                    # Format creation time
                    created_at = session.get("created_at", 0)
                    if isinstance(created_at, (int, float)) and created_at > 0:
                        created_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))
                    else:
                        created_str = "Unknown"
                    
                    # Get token usage
                    tokens = session.get("tokens", {})
                    if isinstance(tokens, dict):
                        total_tokens = sum(tokens.values()) if tokens else 0
                    else:
                        total_tokens = 0
                    
                    # Format token usage
                    token_str = f"{total_tokens:,}" if total_tokens > 0 else "N/A"
                    
                    # Determine tag based on state
                    tag = self.get_state_tag(state)
                    
                    # Create a shortened version of the goal for display
                    short_goal = goal
                    if len(short_goal) > 100:
                        short_goal = short_goal[:97] + "..."
                    
                    # Add row to the table
                    self.tree.insert("", "end", iid=session_id, values=(
                        session_id, 
                        state, 
                        short_goal, 
                        created_str, 
                        token_str
                    ), tags=(tag,))
                    
                    # Store the full session for later
                    self.session_data[session_id] = session
                    
                    # If there's debug data, add a note
                    if session.get("debug_info") or session.get("debug_log"):
                        if self.debug_mode:
                            logger.info(f"Session {session_id} has debug data")
                except Exception as e:
                    logger.error(f"Error processing session item {session.get('session_id', 'unknown')}: {str(e)}", exc_info=True)
            
            # Update status with the total number of sessions
            self.status_label.config(text=f"Loaded {len(sessions)} sessions. Last updated: {time.strftime('%H:%M:%S')}")
            
            # Auto-select the first session if available
            if sessions and not manual:
                children = self.tree.get_children()
                if children:
                    self.tree.selection_set(children[0])
                    self.tree.focus(children[0])
                    self.on_session_select(None)  # Trigger display of the selected session
            
            # Schedule the next refresh if auto-refresh is on
            if self.auto_refresh:
                if self.debug_mode:
                    logger.info(f"Scheduling next refresh in {self.refresh_rate/1000} seconds")
                self.after_handle = self.root.after(self.refresh_rate, self.refresh_data)
        except Exception as e:
            error_msg = f"Error updating sessions table: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.status_label.config(text=error_msg)
    
    def get_state_tag(self, state):
        """Determine the tag for a given session state."""
        if state == "active":
            return "active"
        elif state == "completed":
            return "completed"
        elif state == "stopped":
            return "stopped"
        elif state == "failed":
            return "failed"
        elif state == "error":
            return "error"
        else:
            return "unknown"
    
    def on_close(self):
        """Handle window close event."""
        self.is_closing = True
        # Cancel any pending auto-refresh
        if hasattr(self, 'after_handle'):
            self.root.after_cancel(self.after_handle)
        # Destroy the window
        self.root.destroy()

def show_gui():
    """Launch the Session Monitor GUI."""
    try:
        # Create Tk root
        root = tk.Tk()
        root.title("Co-Scientist Session Monitor")
        root.geometry("1200x800")
        
        # Create and run the application
        app = SessionMonitorGUI(root)
        
        # Start the GUI event loop
        root.mainloop()
    except Exception as e:
        logger.error(f"Error in GUI application: {str(e)}", exc_info=True)
        # If we're running in a terminal, print the error
        print(f"Error starting GUI: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Configure logging
    logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=logging_format)
    
    # Create a file handler
    log_file = "session_monitor.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(logging_format))
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    # Log startup info
    logger.info(f"Starting Co-Scientist Session Monitor GUI v{VERSION}")
    
    # Start the GUI
    show_gui() 