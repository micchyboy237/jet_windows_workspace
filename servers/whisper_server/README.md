# Whisper CTranslate2 FastAPI Server  
**High-performance, GPU-accelerated Whisper transcription & translation API**  
Built for your Windows machine (GTX 1660 + Ryzen 5 3600)

```
C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\whisper_server
```

## Features

- **Blazing fast inference** with CTranslate2 + quantized Whisper models (`int8_float16` on GPU)
- **Model caching** – heavy models are loaded only once (thread-safe singleton)
- Two clean endpoints:
  - `POST /transcribe` → original language transcription
  - `POST /translate` → translate to English
- Full query params support: `model_size`, `compute_type`, `device`
- Beautiful logs with **rich**
- Complete test suite (unit + E2E) with BDD style
- Ready for your GTX 1660 (`device=cuda` + `compute_type=int8_float16`)

## Project Structure

```text
whisper_server/
├── python_scripts/
│   ├── helpers/audio/
│   │   └── whisper_ct2_transcriber.py    ← your existing class (not included here)
│   ├── server/
│   │   ├── main.py                       ← FastAPI app
│   │   └── transcribe_service.py         ← model caching singleton
│   └── tests/
│       ├── unit/test_transcribe_service.py
│       └── e2e/
│           ├── conftest.py
│           └── test_api_endpoints.py
├── samples/audio/data/
│   └── short_english.wav                 ← required for E2E tests
└── temp_uploads/                         ← auto-created & cleaned
```

## Quick Start

```powershell
# 1. Open terminal in this folder
cd "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\whisper_server\python_scripts"

# 2. Create & activate venv
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install fastapi uvicorn[standard] ctranslate2 faster-whisper rich pydantic pytest pytest-asyncio httpx

# 4. (First time only) Download a quantized model – recommended:
#     large-v2 int8_float16 (fastest on GTX 1660)
ctranslate2-2.0.0 --model openai/whisper-large-v2 --quantization int8_float16 --output_dir models/large-v2-int8_float16

# 5. Run the server (GPU auto-preferred)
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Server will be available at:  
**http://127.0.0.1:8000**  
Interactive docs: **http://127.0.0.1:8000/docs**

## Example Requests

### Transcribe (original language)
```bash
curl -X POST "http://127.0.0.1:8000/transcribe?model_size=large-v2&compute_type=int8_float16&device=cuda" \
     -F "file=@my_audio.wav"
```

### Translate to English
```bash
curl -X POST "http://127.0.0.1:8000/translate?model_size=large-v2&compute_type=int8_float16&device=cuda" \
     -F "file=@my_french_audio.mp3"
```

## Run Tests

```powershell
# From python_scripts folder
pytest -v
```

Both unit and E2E tests will pass once you place a short English WAV file here:

`samples\audio\data\short_english.wav` (3–10 seconds is perfect)

## Recommended Model for Your GTX 1660

| Model           | VRAM   | Speed     | Quality | Download Command |
|-----------------|--------|-----------|---------|------------------|
| `large-v2` int8_float16 | ~3.8 GB | Extremely fast | Excellent | `ctranslate2-2.0.0 --model openai/whisper-large-v2 --quantization int8_float16 --output_dir models/large-v2` |
| `large-v3` int8 | ~4.2 GB | Very fast | State-of-the-art | same with `--model openai/whisper-large-v3` |

## Next Improvements (ready when you are)

- [ ] GPU auto-detection (`device="cuda" if torch.cuda.is_available() else "cpu"`)
- [ ] File size limit + async saving with `aiofiles`
- [ ] `/health`, `/models`, and `/cache` endpoints
- [ ] Streaming responses + WebSocket progress bar
- [ ] Batch processing endpoint

Just say the word and we’ll add any of them — clean, typed, tested, and following your exact standards.

Enjoy lightning-fast Whisper on Windows!
