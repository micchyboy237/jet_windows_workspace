# Client Usage Examples

The server runs by default on `http://shawn-pc.local:8001`  
Interactive Swagger UI: <http://shawn-pc.local:8001/docs>

## 1. Health check

```bash
curl http://shawn-pc.local:8001/
```

## 2. Transcribe a local audio file (CTranslate2 – highest quality)

### curl
```bash
curl -X POST "http://shawn-pc.local:8001/transcribe" \
  -F "file=@/path/to/your/audio.wav" \
  -F "model_size=large-v2" \
  -F "compute_type=int8_float16" \
  -F "device=cuda"          # use "cpu" if no GPU
```

### Python (requests)
```python
import requests

url = "http://shawn-pc.local:8001/transcribe"
files = {"file": open("audio.wav", "rb")}
params = {
    "model_size": "large-v2",
    "compute_type": "int8_float16",   # or "int8" on CPU
    "device": "cuda",                 # or "cpu"
}

response = requests.post(url, files=files, params=params)
print(response.json())
```

### Python (httpx – async)
```python
import httpx

async def transcribe():
    async with httpx.AsyncClient() as client:
        with open("audio.wav", "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            params = {
                "model_size": "large-v2",
                "compute_type": "int8_float16",
                "device": "cuda",
            }
            r = await client.post("http://127.0.0.0.1:8001/transcribe", files=files, params=params, timeout=None)
            print(r.json())

import asyncio
asyncio.run(transcribe())
```

## 3. Translate to English (CTranslate2)

Just change the endpoint to `/translate` – everything else stays the same.

```bash
curl -X POST "http://shawn-pc.local:8001/translate" \
  -F "file=@audio_french.mp3" \
  -F "model_size=large-v2" \
  -F "compute_type=int8_float16" \
  -F "device=cuda"
```

Python example is identical except `url = "http://shawn-pc.local:8001/translate"`.

## 4. Streaming transcription (faster-whisper – lower latency, single call)

```bash
curl -X POST "http://shawn-pc.local:8001/transcribe_stream" \
  -F "file=@short_clip.wav" \
  -F "task=transcribe" \
  -F "language=es"           # optional, auto-detect if omitted
```

```python
# Same as normal transcribe but endpoint is /transcribe_stream
response = requests.post(
    "http://shawn-pc.local:8001/transcribe_stream",
    files={"file": open("short_clip.wav", "rb")},
    data={"task": "translate", "language": None},
)
print(response.json()["text"])
```

## 5. Real-time chunk streaming (raw PCM 16kHz float32)

Useful for WebSocket-like pipelines or microphone capture.

```bash
# Example: send first 10 seconds of a 16kHz float32 raw file
curl -X POST "http://shawn-pc.local:8001/transcribe_chunk?task=transcribe" \
  --data-binary @chunk_10sec.raw \
  -H "Content-Type: application/octet-stream"
```

```python
import numpy as np

# Assume `audio_np` is numpy float32 array @ 16kHz
chunk_bytes = audio_np.tobytes()

response = requests.post(
    "http://shawn-pc.local:8001/transcribe_chunk",
    params={"task": "transcribe"},
    data=chunk_bytes,
    headers={"Content-Type": "application/octet-stream"},
)
print(response.json()["text"]
```

## Common query parameters

| Parameter      | Values                          | Description                                      |
|----------------|---------------------------------|--------------------------------------------------|
| `model_size`   | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | Model accuracy vs speed trade-off |
| `compute_type` | `int8`, `int8_float16`, `float16` (GPU only) | Quantization – `int8_float16` recommended for GTX 1660 |
| `device`       | `cpu`, `cuda`                   | Inference device                                 |
| `task`         | `transcribe`, `translate`       | Only for streaming endpoints                     |

Enjoy the blazing-fast Whisper API!