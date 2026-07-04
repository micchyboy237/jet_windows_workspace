# 🎵 Audio Streaming Demo

A minimal audio streaming server and player using **FastAPI** and **HTTP Range Requests** (206 Partial Content). No WebSockets, no complex protocol negotiation — just native browser `<audio>` range support.

## Why This Approach?

| Alternative                  | Why We Skipped It                                       |
| ---------------------------- | ------------------------------------------------------- |
| WebSocket streaming          | Overhead for on-demand files, stateful, harder to cache |
| HLS / DASH                   | Requires transcoding + manifest generation              |
| Web Audio API (decode fully) | High memory usage, no seeking until fully loaded        |
| Chunked transfer (no Range)  | No seeking, browser can't skip ahead                    |

**Our choice:** HTTP `Range` requests with `206 Partial Content` responses  
→ Browser-native seeking, low memory, CDN-cacheable, stateless.

---

## Architecture

```
┌─────────────────┐     Range: bytes=0-524287     ┌──────────────────┐
│                 │ ──────────────────────────────> │                  │
│   Browser       │     206 Partial Content        │   FastAPI        │
│   (Howler.js    │ <────────────────────────────── │   /audio/{file}  │
│    html5:true)  │     Content-Range: bytes       │                  │
│                 │     0-524287/5432101           │   /tracks        │
└─────────────────┘                                └──────────────────┘
```

### Streaming Flow

```
1. Browser loads page → fetches GET /tracks → populates queue
2. User clicks track → Howler creates <audio> element (html5:true)
3. Browser sends: GET /audio/song.mp3 (no Range header)
4. Server returns: 206, bytes 0-524287 (first 512 KB)
5. Audio starts playing immediately from that chunk
6. As playback progresses, browser auto-sends new Range requests
7. On seek: browser sends Range: bytes=<new_pos>- → instant
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn
```

### 3. Add audio files

Place your audio files in the `audio_files/` directory:

```bash
mkdir -p audio_files
# Add your .mp3, .ogg, .flac, .wav, .aac, .m4a, or .opus files here
```

### 4. Start the server

```bash
python main.py
```

Server starts at **http://localhost:8000**

### 5. Open the player

Open `player.html` in your browser, or serve it from the FastAPI server by adding static file serving.

---

## API Endpoints

| Method | Endpoint                 | Description                          |
| ------ | ------------------------ | ------------------------------------ |
| `GET`  | `/tracks`                | List all available audio files       |
| `GET`  | `/audio/{filename}`      | Stream audio with Range support      |
| `HEAD` | `/audio/{filename}`      | Get headers only (duration metadata) |
| `GET`  | `/audio/{filename}/info` | Get file metadata (size, mime type)  |

### Example Responses

**GET /tracks**

```json
{
  "tracks": [
    {
      "filename": "song.mp3",
      "size_bytes": 5432101,
      "mime_type": "audio/mpeg"
    }
  ]
}
```

**GET /audio/song.mp3** (first request, no Range)

```
HTTP/1.1 206 Partial Content
Content-Range: bytes 0-524287/5432101
Accept-Ranges: bytes
Content-Length: 524288
Content-Type: audio/mpeg
```

**GET /audio/song.mp3/info**

```json
{
  "filename": "song.mp3",
  "size_bytes": 5432101,
  "mime_type": "audio/mpeg",
  "size_mb": 5.18
}
```

---

## Player Features

### Controls

- ▶ Play / Pause (click or Space key)
- ⏮ Previous / Next track (P / N keys)
- ⏪ Rewind 10 seconds (← key)
- ⏩ Forward 10 seconds (→ key)
- 🔊 Volume slider + Mute toggle (M key)

### Visual Feedback

- Waveform bars that fill as playback progresses
- Buffer progress indicator (lighter bars + percentage)
- Active track highlighting in queue
- Loading spinner during initial fetch
- Live time display (current / total)

### Queue

- Click any track to play
- Auto-advance to next track on end
- Show/hide toggle

---

## Configuration

| Variable               | Default                 | Description                                   |
| ---------------------- | ----------------------- | --------------------------------------------- |
| `API` (in player.html) | `http://localhost:8000` | FastAPI server URL                            |
| `CHUNK_SIZE`           | 256 KB                  | Streaming chunk size                          |
| `INITIAL_BUFFER`       | 512 KB                  | Bytes sent on first request (no Range header) |

---

## Production Notes

### CORS

The server allows all origins (`*`) for development. Restrict this in production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    ...
)
```

### Supported Audio Formats

`.mp3`, `.ogg`, `.flac`, `.wav`, `.aac`, `.m4a`, `.opus`

### Caching

Responses include `Cache-Control: public, max-age=3600` — adjust for your use case.

### Behind a Reverse Proxy (nginx)

```nginx
proxy_set_header Host $host;
proxy_pass http://127.0.0.1:8000;
proxy_buffering off;  # Important for streaming
```

---

## Project Structure

```
audio_streaming/demo1/
├── main.py           # FastAPI server
├── player.html       # Frontend player (Howler.js + vanilla JS)
├── audio_files/      # Put your audio files here
│   ├── song1.mp3
│   └── podcast.ogg
└── README.md
```

---

## Troubleshooting

| Issue                   | Fix                                                 |
| ----------------------- | --------------------------------------------------- |
| "No server — demo mode" | FastAPI server not running or CORS blocked          |
| Tracks not showing      | Add audio files to `audio_files/` directory         |
| Playback doesn't start  | Check browser console for CORS errors               |
| Seeking doesn't work    | Ensure server returns `Accept-Ranges: bytes` header |
| Mobile autoplay blocked | Player handles this via Howler's `unlock` event     |

---

## License

MIT
