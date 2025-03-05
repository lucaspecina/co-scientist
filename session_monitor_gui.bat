@echo off
echo Starting AI Co-Scientist Session Monitor GUI...
echo.
echo If the application doesn't start, please check the following:
echo 1. Make sure you're running this from the project root directory
echo 2. Ensure Python is installed and in your PATH
echo 3. Check the gui_monitor.log file for errors
echo.

python session_monitor_gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: The GUI application failed to start with error code %ERRORLEVEL%
    echo Please check the gui_monitor.log file for details.
    pause
) 