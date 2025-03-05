@echo off
if "%1"=="" (
    echo Error: Please provide a session ID
    echo Usage: new-blank-session.bat [session_id]
    exit /b 1
)

set SESSION_ID=%1

if not exist data mkdir data
if not exist data\sessions mkdir data\sessions

echo {> data\sessions\%SESSION_ID%.json
echo   "session_id": "%SESSION_ID%",>> data\sessions\%SESSION_ID%.json
echo   "goal": "Test session",>> data\sessions\%SESSION_ID%.json
echo   "state": "initialized",>> data\sessions\%SESSION_ID%.json
echo   "created_at": "%DATE% %TIME%">> data\sessions\%SESSION_ID%.json
echo }>> data\sessions\%SESSION_ID%.json

copy data\sessions\%SESSION_ID%.json data\session_%SESSION_ID%.json > nul

echo Created new blank session with ID: %SESSION_ID%
echo Files created:
echo   - data\sessions\%SESSION_ID%.json
echo   - data\session_%SESSION_ID%.json 