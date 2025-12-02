@echo off
setlocal

:: ===================================================================
::  Convert OpenAI Whisper large-v3 → CTranslate2 int8 (GTX 1660 optimal)
::  Run this file once — creates a permanent, offline model
:: ===================================================================

set "OUTPUT_DIR=C:\asr-models\faster-whisper-large-v3-int8"

echo.
echo ============================================================
echo  Converting openai/whisper-large-v3 → int8 (CUDA optimized)
echo  Output folder: %OUTPUT_DIR%
echo ============================================================
echo.

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Run the actual conversion
ct2-transformers-converter ^
  --model openai/whisper-large-v3 ^
  --output_dir "%OUTPUT_DIR%" ^
  --quantization int8 ^
  --force

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   SUCCESS! Model ready at:
    echo   %OUTPUT_DIR%
    echo.
    echo   You can now use it with:
    echo   WhisperModel(r"%OUTPUT_DIR%", device="cuda", compute_type="int8")
    echo ============================================================
) else (
    echo.
    echo XXX Conversion FAILED — check errors above XXX
)

echo.
echo Press any key to exit...
pause >nul
endlocal
