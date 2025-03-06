# AI Co-Scientist Session Monitor GUI

This GUI application allows you to monitor your AI Co-Scientist sessions in real-time while continuing to use terminal commands independently.

## Features

- Real-time session monitoring with auto-refresh
- Filter sessions by state (active, stopped, completed, failed)
- View detailed session information, including token usage and estimated costs
- Stop individual sessions or all active sessions
- Delete sessions
- Color-coded session states for easy identification
- **Session Log Viewer**: View and monitor logs from sessions in real-time
- **Terminal Output Display**: See terminal outputs and commands as they would appear in the terminal
- **Robust File Access**: Falls back to direct file system access if the memory manager fails

## Usage

### On Windows

You can use the provided batch file:

```
session_monitor_gui.bat
```

Or run the Python script directly:

```
python session_monitor_gui.py
```

### On PowerShell

```
.\Session-Monitor-GUI.ps1
```

Or run the Python script directly:

```
python session_monitor_gui.py
```

### On Linux/macOS

```
python session_monitor_gui.py
```

## Working with Terminal Commands

The GUI application is designed to work independently of terminal commands. This means you can:

1. Run the GUI monitor in one window
2. Execute terminal commands in a separate window
3. See the changes reflected in the GUI automatically (if auto-refresh is enabled)

### Example Workflow

1. Start the GUI monitor application
2. Open a separate terminal window
3. Create a new session via terminal:
   ```
   python create_session.py "My research goal"
   ```
4. Watch the new session appear in the GUI
5. Run the session via terminal:
   ```
   python run_session.py <session_id>
   ```
6. Monitor the session's progress and token usage in the GUI
7. **View the session logs by clicking "View Logs" or switching to the "Session Logs" tab**
8. Provide feedback via terminal if needed:
   ```
   python provide_feedback.py <session_id> "This is interesting, please explore more about X"
   ```
9. Use the GUI to stop the session when done

## Session Log Viewer

The Session Log Viewer tab provides a comprehensive view of all logs associated with a session:

### Available Log Types

- **Session Logs**: The main session execution logs
- **Agent Logs**: Logs from individual agents within the session
- **System Logs**: Filtered system logs related to the session
- **Terminal Output**: Terminal commands and their outputs

### Using the Log Viewer

1. Select a session from the dropdown
2. Choose the log type you want to view
3. Click "Refresh Logs" to update the display
4. Enable "Auto-refresh logs" to automatically update when the session data refreshes

The log viewer will automatically highlight different types of log messages:
- Errors are shown in red
- Warnings in orange
- Info messages in green
- Debug messages in blue
- Commands in purple
- Regular output in black

### Finding Log Files

The application will automatically scan for log files in various common locations:
- The session directory
- Dedicated log directories
- System log files

If no log files are found, the application will try to extract log information directly from the session data object.

## Troubleshooting

### Import Errors

If you see import errors when starting the application, make sure you are running it from the root directory of the AI Co-Scientist project.

### Session Not Appearing

If newly created sessions don't appear in the GUI:
1. Click the "Refresh Now" button to force a refresh
2. Check that the session was actually created successfully
3. Verify that the memory manager is working correctly

### GUI Not Responsive

If the GUI becomes unresponsive:
1. Check the `gui_monitor.log` file for errors
2. Restart the application
3. If the issue persists, there may be issues with the memory manager or database

### Log Files Not Found

If log files aren't showing up:
1. Make sure the session directory structure is correct
2. Check if logs are being written to custom locations
3. Ensure the application has permission to read the log files
4. Try selecting "All" in the log type dropdown to search for all possible log files

## Technical Details

The GUI application is built using:
- Python's Tkinter library for the user interface
- Asynchronous operations for database access
- Threading to prevent UI freezes
- The same underlying session management code used by the terminal commands
- Fallback mechanisms to directly read session data from files if the memory manager fails

The application runs separately from terminal commands but interacts with the same session data storage, allowing for real-time monitoring while you continue to work with terminal commands. 