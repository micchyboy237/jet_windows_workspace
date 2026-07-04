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
import logging
import mimetypes
import sys
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

# ── Logging Setup ────────────────────────────────────────────
logger = logging.getLogger("audio_stream")
logger.setLevel(logging.DEBUG)

# Console handler with formatted output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# Suppress noisy uvicorn/httpx logs unless needed
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(title="Audio Stream API")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming HTTP request and its outcome."""
    logger.info(
        "→ REQUEST  | %s %s | client=%s | user-agent=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
        request.headers.get("user-agent", "unknown")[:80],
    )
    try:
        response = await call_next(request)
        logger.info(
            "← RESPONSE | %s %s → %s | content-length=%s",
            request.method,
            request.url.path,
            response.status_code,
            response.headers.get("content-length", "streaming"),
        )
        return response
    except Exception as exc:
        logger.error(
            "✗ ERROR    | %s %s → %s | %s",
            request.method,
            request.url.path,
            500,
            str(exc),
            exc_info=True,
        )
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["Range", "Content-Type"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"],
)

# ── Configuration ────────────────────────────────────────────
# Multiple audio directories — add as many as you need
# Priority: files in directories listed first take precedence on name collision
AUDIO_DIRS: List[Path] = [
    # Local audio_files directory (same parent as main.py)
    Path(__file__).parent / "audio_files",
    # Remote segment audio directory
    Path(r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\segment_audio"),
]

# Create directories that don't exist
for d in AUDIO_DIRS:
    d.mkdir(exist_ok=True)

CHUNK_SIZE = 256 * 1024       # 256 KB per chunk
INITIAL_BUFFER = 512 * 1024   # 512 KB initial response when no Range header

logger.info("✓ Audio directories configured (%d total):", len(AUDIO_DIRS))
for i, d in enumerate(AUDIO_DIRS):
    logger.info("  [%d] %s (exists=%s)", i, d.absolute(), d.exists())
logger.info(
    "✓ Chunk size: %.0f KB | Initial buffer: %.0f KB",
    CHUNK_SIZE / 1024,
    INITIAL_BUFFER / 1024,
)

# ── Directory Index ──────────────────────────────────────────
# Cache: maps filename → Path (built on startup, refreshed on /tracks call)
# Priority: first directory that contains the file wins
_file_index: Dict[str, Path] = {}


def build_file_index() -> Dict[str, Path]:
    """
    Scan all configured audio directories and build a filename → Path map.
    If the same filename exists in multiple directories, the first one wins.
    
    Returns the index dict (also stored in _file_index module variable).
    """
    logger.info("🔍 Building file index from %d directories...", len(AUDIO_DIRS))
    index: Dict[str, Path] = {}
    supported = {".mp3", ".ogg", ".flac", ".wav", ".aac", ".m4a", ".opus"}
    
    for dir_path in AUDIO_DIRS:
        if not dir_path.exists():
            logger.warning("  ✗ Directory not found, skipping: %s", dir_path.absolute())
            continue
        
        files_found = 0
        files_skipped = 0
        for f in sorted(dir_path.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in supported:
                files_skipped += 1
                continue
            if f.name in index:
                logger.debug(
                    "  ⚠ Duplicate filename '%s' — keeping first found (%s), ignoring (%s)",
                    f.name,
                    index[f.name].parent.name,
                    dir_path.name,
                )
                continue
            index[f.name] = f
            files_found += 1
        
        logger.info(
            "  ✓ %s | %d files indexed, %d skipped",
            dir_path.name,
            files_found,
            files_skipped,
        )
    
    logger.info("  ✓ Total unique files indexed: %d", len(index))
    _file_index.clear()
    _file_index.update(index)
    return _file_index


# Build index on startup
build_file_index()


def get_audio_path(filename: str) -> Path:
    """
    Resolve and validate file path using the file index.
    Searches across all configured directories.
    Raises 404 if file not found in any directory.
    """
    logger.debug("  Resolve path | requested=%s", filename)
    
    # Check cached index first
    if filename in _file_index:
        path = _file_index[filename]
        logger.debug("  ✓ File found in index: %s | dir=%s", filename, path.parent.name)
        return path
    
    # Fallback: scan directories directly (handles files added after index build)
    logger.debug("  File not in index, scanning directories...")
    for dir_path in AUDIO_DIRS:
        candidate = dir_path / Path(filename).name
        if candidate.exists() and candidate.is_file():
            logger.debug(
                "  ✓ File found via fallback scan: %s | dir=%s",
                filename,
                dir_path.name,
            )
            _file_index[filename] = candidate  # update cache
            return candidate
    
    logger.warning(
        "  ✗ File not found in any directory: %s | searched %d dirs",
        filename,
        len(AUDIO_DIRS),
    )
    raise HTTPException(status_code=404, detail=f"File not found: {filename}")


def parse_range_header(range_header: Optional[str], file_size: int) -> tuple[int, int]:
    """
    Parse 'Range: bytes=START-END' header.
    Returns (start, end) byte positions (inclusive).
    Falls back to [0, INITIAL_BUFFER-1] when no Range sent (first load).
    """
    if not range_header:
        default_end = min(INITIAL_BUFFER - 1, file_size - 1)
        logger.debug(
            "  No Range header → default initial chunk | bytes=0-%d/%d",
            default_end,
            file_size,
        )
        return 0, default_end

    logger.debug("  Parsing Range header: %s", range_header)
    try:
        range_str = range_header.replace("bytes=", "")
        parts = range_str.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start < 0:
            logger.error(
                "  ✗ Invalid range: start=%d, end=%d, file_size=%d",
                start,
                end,
                file_size,
            )
            raise ValueError("Invalid range")
        logger.debug(
            "  ✓ Parsed range: bytes=%d-%d/%d (%.1f KB)",
            start,
            end,
            file_size,
            (end - start + 1) / 1024,
        )
        return start, end
    except Exception as exc:
        logger.error("  ✗ Range parse failed: %s | header=%s", str(exc), range_header)
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")


async def stream_file(path: Path, start: int, end: int):
    """
    Async generator that streams file bytes in CHUNK_SIZE blocks.
    Logs streaming progress at intervals.
    """
    total_bytes = end - start + 1
    bytes_streamed = 0
    chunk_count = 0

    logger.debug(
        "  ▶ Streaming start | file=%s | range=%d-%d | total=%d bytes (%.1f KB)",
        path.name,
        start,
        end,
        total_bytes,
        total_bytes / 1024,
    )

    with open(path, "rb") as f:
        f.seek(start)
        while bytes_streamed < total_bytes:
            read_size = min(CHUNK_SIZE, total_bytes - bytes_streamed)
            data = f.read(read_size)
            if not data:
                logger.warning(
                    "  ⚠ Unexpected EOF at byte %d/%d", bytes_streamed, total_bytes
                )
                break
            bytes_streamed += len(data)
            chunk_count += 1
            if chunk_count % 10 == 0 or bytes_streamed >= total_bytes:
                pct = (bytes_streamed / total_bytes * 100) if total_bytes > 0 else 100
                logger.debug(
                    "  ▸ Stream chunk #%d | %d/%d bytes (%.0f%%)",
                    chunk_count,
                    bytes_streamed,
                    total_bytes,
                    pct,
                )
            yield data

    logger.debug(
        "  ■ Streaming complete | %s | %d bytes in %d chunks",
        path.name,
        bytes_streamed,
        chunk_count,
    )

# ── Routes ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def player():
    """
    Serve the audio player HTML page.
    Open http://localhost:8001/ in your browser to use the player.
    """
    logger.info("☐ Serving player HTML page")
    player_html = Path(__file__).parent / "player.html"
    if not player_html.exists():
        logger.error("✗ player.html not found at %s", player_html.absolute())
        raise HTTPException(status_code=404, detail="player.html not found")
    logger.debug("  ✓ player.html loaded (%d bytes)", player_html.stat().st_size)
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
    logger.info(
        "☐ Audio request | file=%s | method=%s | range=%s",
        filename,
        request.method,
        range or "none",
    )

    path = get_audio_path(filename)
    file_size = path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "audio/mpeg"

    logger.debug(
        "  File info | name=%s | dir=%s | size=%d bytes (%.1f MB) | mime=%s",
        filename,
        path.parent.name,
        file_size,
        file_size / 1024 / 1024,
        mime_type,
    )

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

    logger.debug(
        "  Response headers | %s | content-length=%d | code=206",
        headers["Content-Range"],
        content_length,
    )

    if request.method == "HEAD":
        logger.debug("  HEAD request → 206 without body")
        return Response(status_code=206, headers=headers)

    logger.info(
        "  ▶ Streaming %d bytes (%.1f KB) to client",
        content_length,
        content_length / 1024,
    )

    return StreamingResponse(
        stream_file(path, start, end),
        status_code=206,
        headers=headers,
        media_type=mime_type,
    )


@app.get("/audio/{filename}/info")
async def audio_info(filename: str):
    """Returns file metadata (size, duration hint, mime type) for player UI."""
    logger.info("☐ File info request | file=%s", filename)
    path = get_audio_path(filename)
    file_size = path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "audio/mpeg"

    info = {
        "filename": filename,
        "size_bytes": file_size,
        "mime_type": mime_type,
        "size_mb": round(file_size / 1024 / 1024, 2),
        "source_dir": path.parent.name,
    }
    logger.debug("  Info response | %s", info)
    return info


@app.get("/tracks")
async def list_tracks():
    """
    List all available audio files across all configured directories.
    Refreshes the file index to catch newly added files.
    """
    logger.info("☐ Listing tracks across %d directories", len(AUDIO_DIRS))
    
    # Refresh index to pick up new files
    build_file_index()
    
    tracks = []
    for filename, filepath in sorted(_file_index.items()):
        size = filepath.stat().st_size
        mime = mimetypes.guess_type(filename)[0] or "audio/mpeg"
        tracks.append({
            "filename": filename,
            "size_bytes": size,
            "mime_type": mime,
            "source_dir": filepath.parent.name,
        })
        logger.debug(
            "  ✓ Track | %s | %s | %.1f MB | dir=%s",
            filename,
            mime,
            size / 1024 / 1024,
            filepath.parent.name,
        )
    
    logger.info("  ✓ Listed %d tracks total", len(tracks))
    return {"tracks": tracks, "directories": [str(d.absolute()) for d in AUDIO_DIRS]}


@app.get("/directories")
async def list_directories():
    """List all configured audio directories and their status."""
    logger.info("☐ Listing configured directories")
    dirs = []
    for d in AUDIO_DIRS:
        exists = d.exists()
        file_count = sum(1 for f in d.iterdir() if f.is_file()) if exists else 0
        dirs.append({
            "path": str(d.absolute()),
            "exists": exists,
            "file_count": file_count,
        })
        logger.debug("  Dir | %s | exists=%s | files=%d", d.name, exists, file_count)
    return {"directories": dirs}


# ── Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("═══════════════════════════════════════")
    logger.info("  Audio Stream Server starting...")
    logger.info("  → http://localhost:8001/  (player)")
    logger.info("  → http://localhost:8001/tracks (API)")
    logger.info("  → http://localhost:8001/directories (dirs)")
    logger.info("═══════════════════════════════════════")

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
