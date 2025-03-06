# AI Co-Scientist Session Management

This document explains how to use the session management tools to control your AI Co-Scientist sessions and prevent API overspending.

## Overview

The `manage_sessions.py` script provides a comprehensive interface for managing all aspects of your AI Co-Scientist sessions:

- List all sessions (active, stopped, completed, failed)
- View detailed information about specific sessions
- Stop specific or all active sessions
- Delete specific or all sessions
- Monitor API usage and associated costs

## Usage

### On Windows

You can use the provided batch file:

```
manage_sessions.bat <command> [options]
```

Or run the Python script directly:

```
python manage_sessions.py <command> [options]
```

### On Linux/macOS

```
python manage_sessions.py <command> [options]
```

## Available Commands

### List Sessions

List all sessions with basic information:

```
python manage_sessions.py list
```

For a simpler, more reliable listing (recommended):

```
python manage_sessions.py list --simple
```

Or use the provided batch file (from Command Prompt):

```
list_sessions.bat
```

From PowerShell, either use:

```
.\list_sessions.bat
```

Or use the PowerShell script:

```
.\List-Sessions.ps1
```

Filter sessions by state:

```
python manage_sessions.py list --filter active
python manage_sessions.py list --filter stopped
python manage_sessions.py list --filter completed
python manage_sessions.py list --filter failed
```

Show detailed information:

```
python manage_sessions.py list --verbose
```

### View Session Details

Get detailed information about a specific session:

```
python manage_sessions.py details <session_id>
```

This includes:
- Session metadata (goal, state, creation time)
- Token usage and estimated cost
- Breakdown by agent and model
- Top hypotheses

### Stop Sessions

Stop a specific session:

```
python manage_sessions.py stop <session_id>
```

Stop all active sessions:

```
python manage_sessions.py stop-all
```

### Delete Sessions

Delete a specific session:

```
python manage_sessions.py delete <session_id>
```

Delete an active session (use with caution):

```
python manage_sessions.py delete <session_id> --force
```

Delete all sessions:

```
python manage_sessions.py delete-all
```

Delete all sessions including active ones (use with caution):

```
python manage_sessions.py delete-all --force
```

## Preventing API Overspending

To monitor and control your API usage:

1. **Regular monitoring**: Run `python manage_sessions.py list` to see all sessions and their token usage
2. **Stop unnecessary sessions**: Use `python manage_sessions.py stop-all` to stop all active sessions when not in use
3. **Clean up**: Use `python manage_sessions.py delete <session_id>` to remove completed or failed sessions you no longer need
4. **Check details**: Use `python manage_sessions.py details <session_id>` to see detailed token usage by model and agent

## Examples

### Example 1: Monitor active sessions and their costs

```
python manage_sessions.py list --filter active
```

### Example 2: Stop all active sessions to prevent further charges

```
python manage_sessions.py stop-all
```

### Example 3: Get detailed cost breakdown for a specific session

```
python manage_sessions.py details <session_id>
```

### Example 4: Clean up old sessions to reduce storage

```
python manage_sessions.py delete-all --force
```

## Troubleshooting

- If you encounter errors about missing sessions, ensure your Redis database is running correctly
- If token usage seems incorrect, check the session details to see the breakdown by model and agent
- If you can't delete a session, try using the `--force` flag

### Common Issues

#### MongoDB Not Implemented Error

If you see an error like:
```
Error: Session listing is not implemented in the current MongoDB memory manager.
```

This is because the MongoDB implementation is a stub in the current version. The session manager will automatically fall back to using the file-based storage system instead. Your sessions should still be accessible, but they will be stored in the `data` directory rather than in MongoDB.

#### Memory Manager Shutdown Error

If you see an error like:
```
AttributeError: 'MongoDBMemoryManager' object has no attribute 'shutdown'
```

This is a known issue with the current implementation. Try using the `--debug` flag to see more detailed error information:

```
python manage_sessions.py --debug list
```

#### No Sessions Found

If you're not seeing any sessions when listing:

1. Check that you've actually created sessions with the `create_session.py` script
2. Make sure you're looking in the right data directory
3. Try running with the debug flag to see more information:
   ```
   python manage_sessions.py --debug list
   ```

#### Unhashable Type Dict Error

If you see errors like:
```
Error getting status for session {...}: unhashable type: 'dict'
```

This is due to the complex session processing. Use the simplified listing mode instead:
```
python manage_sessions.py list --simple
```

Or use the provided batch file:
```
list_sessions.bat
```

### Getting Help

If you're still having issues:

1. Run the command with the `--debug` flag for more information
2. Check the `co_scientist.log` file for detailed logs
3. Make sure your storage system (Redis/MongoDB/filesystem) is properly configured

# AI Co-Scientist Session Management Utilities

This folder contains various utility scripts to help manage your AI Co-Scientist sessions.

## Quick List Sessions

Lists all sessions found in your file system:

**PowerShell:**
```
.\Quick-List.ps1
```

**Command Prompt:**
```
quick-list.bat
```

**Direct Python:**
```
python quick_list_sessions.py
```

## Examine a Specific Session

To examine the details of a specific session (providing more diagnostics):

**PowerShell:**
```
.\Examine-Session.ps1 [session_id]
```

**Command Prompt:**
```
examine-session.bat [session_id]
```

**Direct Python:**
```
python examine_session.py [session_id]
```

## Create a New Blank Test Session

If you want to create a new blank session for testing:

**PowerShell:**
```
.\New-BlankSession.ps1 [session_id]
```

**Command Prompt:**
```
new-blank-session.bat [session_id]
```

## Common Issues

If you see an error like:
```
(Error reading file) unsupported for
```

This indicates there might be an issue with the format of your session files. Use the examine script to get more details about the problematic session file.

## PowerShell Execution Policy

If you're having trouble running the PowerShell scripts (.ps1 files), you might need to adjust your execution policy. To run these scripts temporarily, you can use:

```
powershell -ExecutionPolicy Bypass -File .\Script-Name.ps1
```

Or to run it directly without changing policy, use this prefix:
```
.\Script-Name.ps1
```

Remember that in PowerShell, you need to use `.\` to run local scripts in the current directory.
