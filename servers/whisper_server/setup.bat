@echo off
setlocal

:: =============================================================================
:: Whisper CTranslate2 Server – Full Setup
:: Run this once to create venv + install deps + download recommended model
:: =============================================================================

cd /d "%~dp0"

echo.
echo [1/4] Creating virtual environment...
python -m venv venv --clear
if errorlevel 1 (
    echo ERROR: Python not found or venv failed.
    pause
    exit /b 1
)

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate venv.
    pause
    exit /b 1
)

echo.
echo [3/4] Installing Python dependencies...
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" ctranslate2 faster-whisper rich pydantic pytest pytest-asyncio httpx torch --extra-index-url https://download.pytorch.org/whl/cu118

echo.
echo [4/4] Downloading recommended quantized model (large-v2 int8_float16) for GTX 1660...
echo    This is ~3.8 GB and gives the best speed/quality on your GPU
mkdir models 2>nul
ct2-opennmt-ctranslate2-converter --model openai/whisper-large-v2 --output_dir models\large-v2-int8_float16 --quantization int8_float16 --force
if errorlevel 1 (
    echo.
    echo WARNING: Model conversion failed. You can still use CPU or download manually later.
    echo          See README.md for alternative models.
) else (
    echo.
    echo SUCCESS: Model ready at models\large-v2-int8_float16
)

echo.
echo =============================================================================
echo Setup complete!
echo.
echo Next: run  start.bat  to launch the server
echo =============================================================================
pause