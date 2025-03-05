Write-Host "Starting AI Co-Scientist Session Monitor GUI..." -ForegroundColor Cyan
Write-Host ""
Write-Host "If the application doesn't start, please check the following:" -ForegroundColor Yellow
Write-Host "1. Make sure you're running this from the project root directory"
Write-Host "2. Ensure Python is installed and in your PATH"
Write-Host "3. Check the gui_monitor.log file for errors"
Write-Host ""

try {
    python session_monitor_gui.py
}
catch {
    Write-Host ""
    Write-Host "Error: The GUI application failed to start with the following error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Please check the gui_monitor.log file for details."
    Read-Host "Press Enter to exit"
} 