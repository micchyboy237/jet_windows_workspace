@echo off
:: start-audio-server.bat
:: Double-click this file or run from CMD/PowerShell to launch the FastAPI audio server with reload

echo.
echo Starting Debug Audio Server (FastAPI + uvicorn)...
echo.

:: Set working directory (same as your cwd in launch.json)
cd /d "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\audio_server"

:: Make sure the project root is in PYTHONPATH so imports work
set PYTHONPATH=C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\audio_server

echo Server will be available at: http://localhost:8001 (or http://0.0.0.0:8001)
echo Press Ctrl+C to stop
echo.

:: Run uvicorn exactly like your debug config
uvicorn python_scripts.server.main:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --reload ^
    --log-level info

:: Keep window open if double-clicked
pause