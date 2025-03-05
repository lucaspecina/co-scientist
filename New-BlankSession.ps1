# PowerShell script to create a new blank session for testing
# Usage: .\New-BlankSession.ps1 [session_id]

param (
    [Parameter(Mandatory=$true)]
    [string]$SessionId
)

# Ensure data directory exists
if (-not (Test-Path -Path "data")) {
    New-Item -Path "data" -ItemType Directory
}

if (-not (Test-Path -Path "data/sessions")) {
    New-Item -Path "data/sessions" -ItemType Directory
}

# Create a blank session with minimal valid structure
$sessionData = @{
    "session_id" = $SessionId
    "goal" = "Test session"
    "state" = "initialized"
    "created_at" = (Get-Date).ToString("o")
} | ConvertTo-Json

# Write to both possible locations
$sessionData | Out-File -FilePath "data/sessions/$SessionId.json" -Encoding utf8
$sessionData | Out-File -FilePath "data/session_$SessionId.json" -Encoding utf8

Write-Host "Created new blank session with ID: $SessionId"
Write-Host "Files created:"
Write-Host "  - data/sessions/$SessionId.json"
Write-Host "  - data/session_$SessionId.json" 