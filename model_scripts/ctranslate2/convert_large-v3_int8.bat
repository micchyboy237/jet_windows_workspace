@echo off
setlocal EnableDelayedExpansion

:: ================================================================
::  Whisper large-v3 → pure int8 (FASTEST on GTX 1660)
::  Full verbose + debug logging
:: ================================================================

set "MODEL=openai/whisper-large-v3"
set "OUTPUT_DIR=C:\asr-models\faster-whisper-large-v3-int8"

echo.
echo ============================================================
echo  Converting %MODEL%  →  pure int8 (GTX 1660 optimal)
echo  Output folder: %OUTPUT_DIR%
echo ============================================================
echo.

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo [VERBOSE] Starting conversion (you will see every layer)...
echo.

ct2-transformers-converter ^
  --model %MODEL% ^
  --output_dir "%OUTPUT_DIR%" ^
  --quantization int8 ^
  --force ^
  --verbose ^
  --log_level debug

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo   SUCCESS! Pure int8 model created (fastest on GTX 1660)
    echo   Location: %OUTPUT_DIR%
    echo.
    echo   Use with:
    echo     compute_type="int8"
    echo ============================================================
) else (
    echo.
    echo XXX CONVERSION FAILED — see verbose log above XXX
)

echo.
echo Press any key to exit...
pause >nul
endlocal