@echo off
setlocal EnableDelayedExpansion

:: ================================================================
::  Whisper large-v3 → int8_float16 (CUDA + Apple Silicon)
::  Verbose logging via environment variables
:: ================================================================

set "MODEL=openai/whisper-large-v3"
set "OUTPUT_DIR=C:\asr-models\faster-whisper-large-v3-int8_float16"

echo.
echo ============================================================
echo  Converting %MODEL%  →  int8_float16 (cross-platform)
echo  Output folder: %OUTPUT_DIR%
echo ============================================================
echo.

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo [INFO] Enabling verbose logging...
set CT2_VERBOSE=1
set TRANSFORMERS_VERBOSITY=debug

echo [VERBOSE] Starting conversion (every layer + details will show)...
echo.

ct2-transformers-converter ^
  --model %MODEL% ^
  --output_dir "%OUTPUT_DIR%" ^
  --quantization int8_float16 ^
  --force

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   SUCCESS! int8_float16 model created
    echo   Location: %OUTPUT_DIR%
    echo.
    echo   Use with:
    echo     compute_type="int8_float16"   (works on GTX 1660 AND Mac M1/M2/M3/M4)
    echo ============================================================
) else (
    echo.
    echo XXX CONVERSION FAILED — see verbose log above XXX
)

echo.
echo Press any key to exit...
pause >nul
endlocal