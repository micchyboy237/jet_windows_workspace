# Client Usage Examples

The server runs by default on `http://127.0.0.1:8001` (or your local network name)  
Interactive Swagger UI: <http://127.0.0.1:8001/docs>

## Recommended Settings for Your Hardware (GTX 1660 + Windows 11)

| Endpoint                     | Model        | compute_type     | device | Notes                              |
|------------------------------|--------------|------------------|--------|------------------------------------|
| `/transcribe`, `/translate`  | `large-v3`   | `int8_float16`   | `cuda` | Best quality + fast on GTX 1660     |
| Streaming / Chunk endpoints  | `large-v3`   | `int8_float16`   | `cuda` | Low latency + high accuracy        |

---

## 1. Health check

```bash
curl http://127.0.0.1:8001/
```

---

## 2. High-Quality Transcription (CTranslate2 – best accuracy)

### cURL

```bash
curl -X POST "http://127.0.0.1:8001/transcribe" \
  -F "file=@audio.wav" \
  -F "model_size=large-v3" \
  -F "compute_type=int8_float16" \
  -F "device=cuda" \
  -F "translate=true"   # optional – also return English translation
```

### Python (requests) – transcription only

```python
import requests

url = "http://127.0.0.1:8001/transcribe"
files = {"file": open("audio.wav", "rb")}
data = {
    "model_size": "large-v3",
    "compute_type": "int8_float16",
    "device": "cuda",
    # "translate": "true"   # uncomment to also get translation
}
resp = requests.post(url, files=files, data=data)
print(resp.json()["transcription"])
```

### Python (requests) – transcription **+** translation in one call

```python
data["translate"] = "true"
resp = requests.post(url, files=files, data=data)
result = resp.json()
print("Transcription:", result["transcription"])
print("Translation:  ", result.get("translation"))
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
                "translate": "true",
            }
            r = await client.post("http://127.0.0.1:8001/transcribe", files=files, data=data)
            print(r.json()["transcription"])
            print(r.json().get("translation"))

asyncio.run(transcribe())
```

---

## 3. High-Quality Translation of Audio Directly to English

Just change the endpoint – everything else stays the same.

```bash
curl -X POST "http://127.0.0.1:8001/translate" \
  -F "file=@audio_french.mp3" \
  -F "model_size=large-v3" \
  -F "compute_type=int8_float16" \
  -F "device=cuda"
```

```python
resp = requests.post(
    "http://127.0.0.1:8001/translate",
    files={"file": open("audio_french.mp3", "rb")},
    data={
        "model_size": "large-v3",
        "compute_type": "int8_float16",
        "device": "cuda",
    },
)
print(resp.json()["translation"])
```

---

## 4. Text-Only Translation (no audio file needed)

Perfect when you already have a transcription.

### cURL

```bash
curl -X POST "http://127.0.0.1:8001/translate/" \
  -H "Content-Type: application/json" \
  -d '{"text": "今日はいい天気ですね。一緒に散歩しませんか？"}'
```

### Python (requests)

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8001/translate/",
    json={"text": "今日はいい天気ですね。一緒に散歩しませんか？"},
    params={"device": "cuda"}  # or "cpu"
)

result = resp.json()
print("Original:   ", result["original"])
print("Translation:", result["translation"])
```

**Expected output:**
```
Original:    今日はいい天気ですね。一緒に散歩しませんか？
Translation: It's nice weather today. Would you like to go for a walk together?

---

## 5. Low-Latency Streaming Transcription (faster-whisper)

```bash
curl -X POST "http://127.0.0.1:8001/transcribe_stream" \
  -F "file=@short_clip.wav" \
  -F "task=transcribe" \
  -F "model_size=large-v3" \
  -F "compute_type=int8_float16" \
  -F "device=cuda"
```

```python
resp = requests.post(
    "http://127.0.0.1:8001/transcribe_stream",
    files={"file": open("short_clip.wav", "rb")},
    data={
        "task": "transcribe",
        "model_size": "large-v3",
        "compute_type": "int8_float16",
        "device": "cuda",
    },
)
print(resp.json()["text"])
```

---

## 6. Real-time Chunk Streaming (raw PCM 16kHz float32)

### From Python (librosa helper)

```python
import librosa
import numpy as np
import requests

def audio_to_raw_bytes(path: str) -> bytes:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    return np.asarray(audio, dtype=np.float32).tobytes()

chunk = audio_to_raw_bytes("short_clip.wav")
resp = requests.post(
    "http://127.0.0.1:8001/transcribe_chunk?task=transcribe",
    data=chunk,
    headers={"Content-Type": "application/octet-stream"},
    timeout=None,
)
result = resp.json()
print("Text:", result["text"])
print("Lang:", result["language"], f"({result['language_probability']:.2f})")
```

### One-liner with ffmpeg

```bash
ffmpeg -i short_clip.wav -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
curl -X POST "http://127.0.0.1:8001/transcribe_chunk?task=transcribe" \
  --data-binary @- \
  -H "Content-Type: application/octet-stream"
```

> Use `task=translate` for direct translation of non-English chunks.

---

Enjoy blazing-fast, production-ready Whisper transcription & translation!