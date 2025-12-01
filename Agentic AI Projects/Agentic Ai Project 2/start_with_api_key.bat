@echo off
echo ========================================
echo   Multi-Agent Airlines System
echo   Starting with API Key...
echo ========================================
echo.

cd /d "%~dp0"

REM Set environment variables
set OPENAI_API_KEY=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
set DATABASE_URL=sqlite:///./airlines.db
set PORT=8001

echo Environment configured:
echo   - OpenAI API Key: Set
echo   - Database: SQLite
echo   - Port: 8001
echo.

echo Starting server...
echo.
echo Open in browser:
echo   UI:  http://localhost:8001/ui
echo   API: http://localhost:8001/docs
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

python run.py

pause

