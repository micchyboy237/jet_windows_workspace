@echo off
setlocal EnableDelayedExpansion

:: =============================================================================
:: Whisper CTranslate2 FastAPI Server – Start
:: =============================================================================

:: Go to the folder where this batch file lives
cd /d "%~dp0"

:: Optional: activate virtual environment (uncomment if you use one)
:: if exist "venv\Scripts\activate.bat" (
::     call venv\Scripts\activate.bat
:: ) else (
::     echo.
::     echo WARNING: Virtual environment not found. Continuing without activation...
::     echo.
:: )

echo.
echo ========================================================
echo  Whisper CTranslate2 FastAPI Server
echo  Starting on http://127.0.0.1:8000
echo  GPU: CUDA + int8_float16 (GTX 1660 optimized)
echo ========================================================
echo.
echo Open the interactive docs at:
echo     http://127.0.0.1:8000/docs
echo.

:: Run uvicorn with python_scripts as the module root
uvicorn server.main:app ^
    --app-dir python_scripts ^
    --host 0.0.0.0 ^
    --port 8000 ^
    --reload ^
    --log-level info

echo.
echo Server stopped.
pause