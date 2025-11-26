@echo off
cd /d %~dp0

@REM :: Activate virtual environment (adjust path if venv is named differently)
@REM if exist "jet_venv\Scripts\activate.bat" (
@REM     call jet_venv\Scripts\activate.bat
@REM ) else if exist "venv\Scripts\activate.bat" (
@REM     call venv\Scripts\activate.bat
@REM ) else (
@REM     echo Virtual environment not found!
@REM     pause
@REM     exit /b 1
@REM )

uvicorn main:app --reload --host 0.0.0.0 --port 8000