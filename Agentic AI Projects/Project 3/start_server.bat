@echo off
echo ========================================
echo   Multi-Agent Airlines System
echo   Starting Server...
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python...
python --version
echo.

echo Starting server on http://localhost:8001
echo.
echo Open in browser:
echo   UI:  http://localhost:8001/ui
echo   API: http://localhost:8001/docs
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

set PORT=8001
python run.py

pause


