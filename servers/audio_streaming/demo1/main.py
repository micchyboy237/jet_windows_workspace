"""
FastAPI Audio Streaming Backend
================================
Strategy: HTTP Range Requests (206 Partial Content)
Why this over alternatives:
- Native browser <audio> range support → simplest stack
- Enables seeking without re-downloading from start
- Stateless, cacheable, CDN-friendly
- No WebSocket overhead for on-demand files
- StreamingResponse in chunks → low memory, starts fast
For LIVE streams: switch to chunked StreamingResponse without Content-Length.
"""

import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

app = FastAPI(title="Audio Stream API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["Range", "Content-Type"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"],
)

AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 256 * 1024
INITIAL_BUFFER = 512 * 1024


def get_audio_path(filename: str) -> Path:
    """Resolve and validate path (prevent directory traversal)."""
    safe = AUDIO_DIR / Path(filename).name
    if not safe.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return safe


def parse_range_header(range_header: Optional[str], file_size: int) -> tuple[int, int]:
    """
    Parse 'Range: bytes=START-END' header.
    Returns (start, end) byte positions (inclusive).
    Falls back to [0, INITIAL_BUFFER-1] when no Range sent (first load).
    """
    if not range_header:
        return 0, min(INITIAL_BUFFER - 1, file_size - 1)

    try:
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start < 0:
            raise ValueError("Invalid range")
        return start, end
    except Exception:
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")


async def stream_file(path: Path, start: int, end: int):
    """Async generator that streams file bytes in CHUNK_SIZE blocks."""
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            read_size = min(CHUNK_SIZE, remaining)
            data = f.read(read_size)
            if not data:
                break
            yield data
            remaining -= len(data)


# ═══════════════════════════════════════════════════════════════
# Player Route — serves the HTML page
# ═══════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def player():
    """
    Serve the audio player HTML page.
    Open http://localhost:8000/ in your browser to use the player.
    """
    player_html = Path(__file__).parent / "player.html"
    if not player_html.exists():
        raise HTTPException(status_code=404, detail="player.html not found")
    return HTMLResponse(content=player_html.read_text(encoding="utf-8"))


@app.get("/audio/{filename}")
@app.head("/audio/{filename}")
async def audio_stream(
    filename: str,
    request: Request,
    range: Optional[str] = Header(default=None),
):
    """
    206 Partial Content audio endpoint.
    Browser flow:
    1. First request: no Range header → server returns initial 512 KB chunk
       (status 206, Content-Range: bytes 0-524287/total)
    2. Browser starts playing immediately from those bytes
    3. Browser sends subsequent Range requests as playback progresses
    4. On seek: browser sends Range: bytes=<seek_pos>- → new 206 response
    5. Each response includes Accept-Ranges: bytes so browser knows to range-request
    HEAD requests are handled for duration metadata without body.
    """
    path = get_audio_path(filename)
    file_size = path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "audio/mpeg"

    start, end = parse_range_header(range, file_size)
    content_length = end - start + 1

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": mime_type,
        "X-Accel-Buffering": "no",
        "Cache-Control": "public, max-age=3600",
    }

    if request.method == "HEAD":
        return Response(status_code=206, headers=headers)

    return StreamingResponse(
        stream_file(path, start, end),
        status_code=206,
        headers=headers,
        media_type=mime_type,
    )


@app.get("/audio/{filename}/info")
async def audio_info(filename: str):
    """Returns file metadata (size, duration hint, mime type) for player UI."""
    path = get_audio_path(filename)
    file_size = path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(path))
    return {
        "filename": filename,
        "size_bytes": file_size,
        "mime_type": mime_type or "audio/mpeg",
        "size_mb": round(file_size / 1024 / 1024, 2),
    }


@app.get("/tracks")
async def list_tracks():
    """List all available audio files."""
    supported = {".mp3", ".ogg", ".flac", ".wav", ".aac", ".m4a", ".opus"}
    tracks = [
        {
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "mime_type": mimetypes.guess_type(f.name)[0] or "audio/mpeg",
        }
        for f in sorted(AUDIO_DIR.iterdir())
        if f.suffix.lower() in supported
    ]
    return {"tracks": tracks}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
