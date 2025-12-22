@echo off
REM start.bat - Windows batch file to easily set up and run the Live Subtitle Server
REM Place this file in the project root (same level as 'server/' and 'client/')

REM Optional: Uncomment the following lines if you want automatic venv creation/activation
REM if not exist venv (
REM     echo Creating virtual environment...
REM     python -m venv venv
REM )
REM call venv\Scripts\activate

echo.
echo ==================================================
echo     Live Subtitle Server - Startup Script
echo ==================================================
echo.

@REM echo [INFO] Installing/Updating required packages...
@REM pip install --upgrade fastapi uvicorn faster-whisper torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
@REM pip install --upgrade websockets sounddevice rich numpy tqdm

echo.
echo [INFO] Packages installed/updated.
echo.

echo [INFO] Starting the server...
echo         Endpoint: ws://<your-ip>:8000/ws/subtitles
echo         Use client/live_subtitle_client.py to connect
echo.
uvicorn server.app:app --host 0.0.0.0 --port 8002 --workers 1 --reload

echo.
echo [ERROR] Server stopped unexpectedly.
pause