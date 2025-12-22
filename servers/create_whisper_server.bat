@echo off
setlocal EnableDelayedExpansion

:: =============================================================================
:: Fixed & Escaped Version - Creates whisper_server in the exact location
:: =============================================================================

set "TARGET=C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\whisper_server"

echo Creating Whisper server at:
echo %TARGET%
echo.

mkdir "%TARGET%" 2>nul
mkdir "%TARGET%\python_scripts\helpers\audio"
mkdir "%TARGET%\python_scripts\server"
mkdir "%TARGET%\python_scripts\tests\unit"
mkdir "%TARGET%\python_scripts\tests\e2e"
mkdir "%TARGET%\samples\audio\data"

type nul > "%TARGET%\python_scripts\helpers\audio\__init__.py"

echo Writing files...

:: 1. transcribe_service.py
(
echo from __future__ import annotations
echo from pathlib import Path
echo from typing import Dict
echo from concurrent.futures import ThreadPoolExecutor
echo from python_scripts.helpers.audio.whisper_ct2_transcriber import WhisperCT2Transcriber, QuantizedModelSizes
echo.
echo _MODEL_CACHE: Dict[str, WhisperCT2Transcriber] = {}
echo _CACHE_LOCK = ThreadPoolExecutor^(max_workers=1^)
echo.
echo def get_transcriber^(
echo     model_size: QuantizedModelSizes = "large-v2",
echo     compute_type: str = "int8_float16",
echo     device: str = "cpu",
echo ^) -^> WhisperCT2Transcriber:
echo     key = f"{model_size}|{compute_type}|{device}"
echo     if key not in _MODEL_CACHE:
echo         def init^(^):
echo             if key not in _MODEL_CACHE:
echo                 _MODEL_CACHE[key] = WhisperCT2Transcriber^(
echo                     model_size=model_size,
echo                     device=device,
echo                     compute_type=compute_type,
echo                 ^)
echo         _CACHE_LOCK.submit^(init^).result^(^)
echo     return _MODEL_CACHE[key]
) > "%TARGET%\python_scripts\server\transcribe_service.py"

:: 2. main.py (fully escaped)
(
echo from __future__ import annotations
echo import logging
echo from pathlib import Path
echo from fastapi import FastAPI, File, UploadFile, HTTPException, Query
echo from pydantic import BaseModel
echo from typing import Optional
echo from .transcribe_service import get_transcriber
echo from python_scripts.helpers.audio.whisper_ct2_transcriber import QuantizedModelSizes
echo from rich.logging import RichHandler
echo.
echo logging.basicConfig^(level=logging.INFO, format="%%^(message^)s", datefmt="[^%%X]", handlers=[RichHandler^(rich_tracebacks=True^)]^)
echo log = logging.getLogger^("whisper-api"^)
echo.
echo app = FastAPI^(title="Whisper CTranslate2 FastAPI Server", version="1.0.0"^)
echo.
echo class TranscriptionResponse^(BaseModel^):
echo     audio_path: str
echo     duration_sec: float
echo     detected_language: Optional[str] = None
echo     detected_language_prob: Optional[float] = None
echo     transcription: str
echo     translation: Optional[str] = None
echo.
echo @app.post^("/transcribe", response_model=TranscriptionResponse^)
echo async def transcribe_audio^(
echo     file: UploadFile = File^(...^),
echo     model_size: QuantizedModelSizes = Query^("large-v2"^),
echo     compute_type: str = Query^("int8_float16"^),
echo     device: str = Query^("cpu"^),
echo ^):
echo     if not file.filename.lower^(^).endswith^(^(".wav",".mp3",".m4a",".flac",".ogg"^)^):
echo         raise HTTPException^(400, "Unsupported file format"^)
echo     content = await file.read^(^)
echo     tmp = Path^("temp_uploads"^) / file.filename
echo     tmp.parent.mkdir^(parents=True, exist_ok=True^)
echo     tmp.write_bytes^(content^)
echo     try:
echo         t = get_transcriber^(model_size, compute_type, device^)
echo         result = t^(tmp, detect_language=True, translate_to_english=False^)
echo         log.info^(f"[Transcribe] {result['detected_language']} {result['duration_sec']:.1f}s"^)
echo         return TranscriptionResponse^(**result^)
echo     finally:
echo         if tmp.exists^(^): tmp.unlink^(^)
echo.
echo @app.post^("/translate", response_model=TranscriptionResponse^)
echo async def translate_audio^(
echo     file: UploadFile = File^(...^),
echo     model_size: QuantizedModelSizes = Query^("large-v2"^),
echo     compute_type: str = Query^("int8_float16"^),
echo     device: str = Query^("cpu"^),
echo ^):
echo     if not file.filename.lower^(^).endswith^(^(".wav",".mp3",".m4a",".flac",".ogg"^)^):
echo         raise HTTPException^(400, "Unsupported file format"^)
echo     content = await file.read^(^)
echo     tmp = Path^("temp_uploads"^) / file.filename
echo     tmp.parent.mkdir^(parents=True, exist_ok=True^)
echo     tmp.write_bytes^(content^)
echo     try:
echo         t = get_transcriber^(model_size, compute_type, device^)
echo         result = t^(tmp, detect_language=True, translate_to_english=True^)
echo         log.info^(f"[Translate] {result['detected_language']} -^> en"^)
echo         return TranscriptionResponse^(**result^)
echo     finally:
echo         if tmp.exists^(^): tmp.unlink^(^)
echo.
echo @app.get^("/"^)
echo async def root^(^): return {"message": "Whisper CTranslate2 API ready"}
) > "%TARGET%\python_scripts\server\main.py"

:: 3–5. Tests (shortened but fully functional & escaped)
(
echo import pytest
echo from server.transcribe_service import get_transcriber
echo from python_scripts.helpers.audio.whisper_ct2_transcriber import QuantizedModelSizes
echo def test_get_transcriber_returns_same_instance^(^):
echo     t1 = get_transcriber^("small","int8","cpu"^)
echo     t2 = get_transcriber^("small","int8","cpu"^)
echo     assert t1 is t2
echo def test_get_transcriber_different_configs_different_instances^(^):
echo     t1 = get_transcriber^("base","int8","cpu"^)
echo     t2 = get_transcriber^("base","int8_float16","cpu"^)
echo     assert t1 is not t2
) > "%TARGET%\python_scripts\tests\unit\test_transcribe_service.py"

(
echo import pytest, shutil
echo from pathlib import Path
echo from fastapi.testclient import TestClient
echo from server.main import app
echo @pytest.fixture^(scope="session"^)
echo def test_audio_sample^(^):
echo     p = Path^("samples/audio/data/short_english.wav"^)
echo     if not p.exists^(^): pytest.skip^("Add short_english.wav"^)
echo     return str^(p^)
echo @pytest.fixture
echo def client^(^):
echo     with TestClient^(app^) as c:
echo         yield c
echo     shutil.rmtree^("temp_uploads", ignore_errors=True^)
) > "%TARGET%\python_scripts\tests\e2e\conftest.py"

(
echo import pytest
echo def test_transcribe_endpoint^(client, test_audio_sample^):
echo     with open^(test_audio_sample, "rb"^) as f:
echo         r = client.post^("/transcribe?model_size=base^&compute_type=int8", files={"file": ^("test.wav", f, "audio/wav"^)}^)
echo     assert r.status_code == 200
echo     j = r.json^(^)
echo     assert j["detected_language"] == "^<|en|^>"
echo     assert len^(j["transcription"]^) ^> 10
echo.
echo def test_translate_endpoint_returns_translation^(client, test_audio_sample^):
echo     with open^(test_audio_sample, "rb"^) as f:
echo         r = client.post^("/translate?model_size=base^&compute_type=int8", files={"file": ^("test.wav", f, "audio/wav"^)}^)
echo     assert r.status_code == 200
echo     assert r.json^(^)["translation"] is not None
) > "%TARGET%\python_scripts\tests\e2e\test_api_endpoints.py"

echo.
echo [SUCCESS] All files created successfully!
echo Location: %TARGET%
echo.
echo Now:
echo   1. Add a short English WAV ^(3-10 sec^) to:
echo      %TARGET%\samples\audio\data\short_english.wav
echo   2. cd %TARGET%\python_scripts
echo   3. python -m venv venv ^& venv\Scripts\activate
echo   4. pip install fastapi uvicorn[standard] ctranslate2 faster-whisper rich pydantic pytest pytest-asyncio httpx
echo   5. uvicorn server.main:app --reload
echo.
pause