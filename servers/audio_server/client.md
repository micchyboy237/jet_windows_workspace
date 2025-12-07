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
            r = await client.post("http://shawn-pc.local:8001/transcribe", files=files, params=params, timeout=None)
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

Useful for microphone capture or processing audio in chunks.

The endpoint expects **raw little-endian 16kHz float32** audio bytes  
(no WAV header, no container).

### From a pre-loaded NumPy array (advanced / microphone)
```python
import numpy as np
import requests

# Example: audio_np is float32, 16kHz, shape (N,)
chunk_bytes = audio_np.tobytes()

response = requests.post(
    "http://shawn-pc.local:8001/transcribe_chunk",
    params={"task": "transcribe"},
    data=chunk_bytes,
    headers={"Content-Type": "application/octet-stream"},
    timeout=None,
)
print(response.json()["text"])
```

### Recommended: From any audio file (WAV, MP3, etc.) – easiest for testing
```python
import librosa
import requests
import numpy as np

def load_audio_for_chunk(file_path: str, sr: int = 16000) -> bytes:
    """Load audio file and return raw 16kHz float32 bytes expected by /transcribe_chunk"""
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    audio = audio.astype(np.float32)
    return audio.tobytes()

# Usage
file_path = "short_clip.wav"  # or .mp3, .m4a, .flac, etc.
chunk_bytes = load_audio_for_chunk(file_path)

response = requests.post(
    "http://shawn-pc.local:8001/transcribe_chunk",
    params={"task": "transcribe"},
    data=chunk_bytes,
    headers={"Content-Type": "application/octet-stream"},
    timeout=None,
)
result = response.json()
print("Text:", result["text"])
print("Language:", result["language"], f"({result['language_probability']:.2f})")
print("Duration:", f"{result['duration_sec']:.2f}s")
```

### Async version (httpx)
```python
import httpx
import librosa
import numpy as np

async def transcribe_chunk_file(file_path: str):
    audio, _ = librosa.load(file_path, sr=16000, mono=True)
    audio_bytes = np.asarray(audio, dtype=np.float32).tobytes()

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            "http://shawn-pc.local:8001/transcribe_chunk?task=transcribe",
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        print(r.json()["text"])

# Run it
import asyncio
asyncio.run(transcribe_chunk_file("short_clip.wav"))
```

### cURL (from file → raw bytes via tool)
```bash
# Using ffmpeg to extract raw float32 16kHz (works on Windows/Linux/macOS)
ffmpeg -i short_clip.wav -f f32le -acodec pcm_f32le -ar 16000 -ac 1 - | \
curl -X POST "http://shawn-pc.local:8001/transcribe_chunk?task=transcribe" \
  --data-binary @- \
  -H "Content-Type: application/octet-stream"
```

> Tip: Use `task=translate` to translate non-English audio directly in streaming mode.

## Common query parameters

| Parameter      | Values                          | Description                                      |
|----------------|---------------------------------|--------------------------------------------------|
| `model_size`   | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | Model accuracy vs speed trade-off |
| `compute_type` | `int8`, `int8_float16`, `float16` (GPU only) | Quantization – `int8_float16` recommended for GTX 1660 |
| `device`       | `cpu`, `cuda`                   | Inference device                                 |
| `task`         | `transcribe`, `translate`       | Only for streaming endpoints                     |

Enjoy the blazing-fast Whisper API!