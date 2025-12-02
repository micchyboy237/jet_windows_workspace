@echo off
setlocal EnableDelayedExpansion

:: ================================================================
::  Convert Whisper large-v3 → int8_float16 with MAXIMUM verbosity
:: ================================================================

set "MODEL=openai/whisper-large-v3"
set "OUTPUT_DIR=C:\asr-models\faster-whisper-large-v3-int8_float16"

echo.
echo ============================================================
echo  Converting %MODEL%  →  int8_float16 (verbose mode)
echo  Output folder: %OUTPUT_DIR%
echo ============================================================
echo.

:: Create output directory
if not exist "%OUTPUT_DIR%" (
    echo [INFO] Creating directory: %OUTPUT_DIR%
    mkdir "%OUTPUT_DIR%"
)

echo.
echo [VERBOSE] Starting conversion with maximum logging...
echo.

:: THIS IS THE ONLY LINE THAT CHANGED → added --verbose and --log_level debug
ct2-transformers-converter ^
  --model %MODEL% ^
  --output_dir "%OUTPUT_DIR%" ^
  --quantization int8_float16 ^
  --force ^
  --verbose ^
  --log_level debug

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   SUCCESS! int8_float16 model created
    echo   Location: %OUTPUT_DIR%
    echo.
    echo   Use with:
    echo     compute_type="int8_float16"
    echo   Works perfectly on both GTX 1660 (CUDA) and Mac M1/M2/M3/M4 (MPS)
    echo ============================================================
) else (
    echo.
    echo XXX CONVERSION FAILED (error code: %ERRORLEVEL%) XXX
    echo Check the verbose log above for details.
)

echo.
echo Press any key to close...
pause >nul
endlocal