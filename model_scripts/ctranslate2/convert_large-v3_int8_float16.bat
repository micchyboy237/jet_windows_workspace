@echo off
setlocal EnableDelayedExpansion

:: ================================================================
::  Whisper large-v3 → int8_float16 (CUDA + Apple Silicon)
::  Saves to your custom cache: C:\Users\druiv\.cache\hf_ctranslate2_models\
:: ================================================================

set "MODEL=openai/whisper-large-v3"
set "BASE_DIR=C:\Users\druiv\.cache\hf_ctranslate2_models"
set "OUTPUT_DIR=%BASE_DIR%\faster-whisper-large-v3-int8_float16"

echo.
echo ============================================================
echo  Converting %MODEL%  →  int8_float16 (cross-platform)
echo  Output folder: %OUTPUT_DIR%
echo ============================================================
echo.

:: Create base + model directory
if not exist "%BASE_DIR%" mkdir "%BASE_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo [INFO] Verbose logging enabled...
set CT2_VERBOSE=1
set TRANSFORMERS_VERBOSITY=debug

echo [INFO] Starting conversion...
echo.

ct2-transformers-converter ^
  --model %MODEL% ^
  --output_dir "%OUTPUT_DIR%" ^
  --quantization int8_float16 ^
  --force

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   SUCCESS! int8_float16 model ready
    echo   Location: %OUTPUT_DIR%
    echo.
    echo   Use with:
    echo     compute_type="int8_float16"   (GTX 1660 + Mac M1/M2/M3/M4)
    echo ============================================================
) else (
    echo.
    echo XXX CONVERSION FAILED — see log above XXX
)

echo.
echo Press any key to exit...
pause >nul
endlocal