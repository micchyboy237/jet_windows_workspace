# Client Usage Examples

The server runs by default on `http://127.0.0.1:8001` (or your local network name)

Interactive Swagger UI: <http://127.0.0.1:8001/docs>

## Recommended Settings for Your Hardware (GTX 1660 + Windows 11)

| Endpoint             | Model           | compute_type     | device | Notes                             |
|----------------------|-----------------|------------------|--------|-----------------------------------|
| `/transcribe`, `/translate` | `large-v3`   | `int8_float16`   | `cuda` | Best quality + fast on GTX 1660   |
| Streaming endpoints  | `large-v3`      | `int8_float16`   | `cuda` | Low latency + high accuracy       |

---

## 1. Health check

```bash
curl http://127.0.0.1:8001/
```

---

## 2. High-Quality Transcription (CTranslate2 – best accuracy)

### cURL (Recommended for GTX 1660)
```bash
curl -X POST "http://127.0.0.1:8001/transcribe" \
  -F "file=@audio.wav" \
  -F "model_size=large-v3" \
  -F "compute_type=int8_float16" \
  -F "device=cuda"
```

### Python (requests)
```python
import requests

url = "http://127.0.0.1:8001/transcribe"
files = {"file": open("audio.wav", "rb")}
data = {
    "model_size": "large-v3",
    "compute_type": "int8_float16",
    "device": "cuda"
}
response = requests.post(url, files=files, data=data)
print(response.json()["transcription"])
```

### Python (httpx – async)
```python
import httpx, asyncio

async def transcribe():
    async with httpx.AsyncClient(timeout=None) as client:
        with open("audio.wav", "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            data = {
                "model_size": "large-v3",
                "compute_type": "int8_float16",
                "device": "cuda",
            }
            r = await client.post("http://127.0.0.1:8001/transcribe", files=files, data=data)
            print(r.json()["transcription"])

asyncio.run(transcribe())
```

---

## 3. Translate to English (CTranslate2)

Same as above, just change endpoint:

```bash
curl -X POST "http://127.0.0.1:8001/translate" \
  -F "file=@audio_french.mp3" \
  -F "model_size=large-v3" \
  -F "compute_type=int8_float16" \
  -F "device=cuda"
```

### Python (requests)
```python
import requests

url = "http://127.0.0.1:8001/translate"
files = {"file": open("audio_french.mp3", "rb")}
data = {
    "model_size": "large-v3",
    "compute_type": "int8_float16",
    "device": "cuda"
}
response = requests.post(url, files=files, data=data)
print(response.json()["translation"])
```

### Python (httpx – async)
```python
import httpx, asyncio

async def translate():
    async with httpx.AsyncClient(timeout=None) as client:
        with open("audio_french.mp3", "rb") as f:
            files = {"file": ("audio_french.mp3", f, "audio/mp3")}
            data = {
                "model_size": "large-v3",
                "compute_type": "int8_float16",
                "device": "cuda",
            }
            r = await client.post("http://127.0.0.1:8001/translate", files=files, data=data)
            print(r.json()["translation"])

asyncio.run(translate())
```
---

## 4. Low-Latency Streaming (faster-whisper)

```bash
curl -X POST "http://127.0.0.1:8001/transcribe_stream" \
  -F "file=@short_clip.wav" \
  -F "task=transcribe" \
  -F "model_size=large-v3" \
  -F "compute_type=int8_float16" \
  -F "device=cuda"
```

```python
response = requests.post(
    "http://127.0.0.1:8001/transcribe_stream",
    files={"file": open("short_clip.wav", "rb")},
    data={
        "task": "transcribe",
        "model_size": "large-v3",
        "compute_type": "int8_float16",
        "device": "cuda",
    },
)
print(response.json()["text"])
```

---

## 5. Real-time Chunk Streaming (raw PCM 16kHz float32)

Perfect for live microphone or chunked processing.

### From any audio file (easiest testing)
```python
import librosa
import numpy as np
import requests

def audio_to_raw_bytes(path: str) -> bytes:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return np.asarray(audio, dtype=np.float32).tobytes()

chunk = audio_to_raw_bytes("short_clip.wav")

response = requests.post(
    "http://127.0.0.1:8001/transcribe_chunk?task=transcribe",
    data=chunk,
    headers={"Content-Type": "application/octet-stream"},
    timeout=None,
)
result = response.json()
print("Text:", result["text"])
print("Lang:", result["language"], f"({result['language_probability']:.2f})")
```

### cURL + ffmpeg (one-liner)
```bash
ffmpeg -i short_clip.wav -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
curl -X POST "http://127.0.0.1:8001/transcribe_chunk?task=transcribe" \
  --data-binary @- \
  -H "Content-Type: application/octet-stream"
```

> Use `task=translate` to translate non-English chunks directly.

---

Enjoy blazing-fast, production-ready Whisper transcription!