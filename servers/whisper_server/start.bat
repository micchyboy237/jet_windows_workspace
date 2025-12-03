@echo off
setlocal EnableDelayedExpansion
:: =============================================================================
:: Whisper CTranslate2 FastAPI Server ΓÇô Start
:: =============================================================================

cd /d "%~dp0"

echo.
echo ========================================================
echo  Whisper CTranslate2 FastAPI Server
echo  Starting on http://127.0.0.1:8001
echo  GPU: CUDA + int8_float16 (GTX 1660 optimized)
echo ========================================================
echo.
echo Open the interactive docs at:
echo     http://127.0.0.1:8001/docs
echo.

:: Now run uvicorn from within the package
uvicorn python_scripts.server.main:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --reload ^
    --log-level info

echo.
echo Server stopped.
cd ..