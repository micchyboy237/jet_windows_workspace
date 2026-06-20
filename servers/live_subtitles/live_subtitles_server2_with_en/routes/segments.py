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
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.state import get_speaker_labeler
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from services.audio_config import SAMPLE_RATE
from services.audio_utils import resolve_audio_paths
from services.config import SEGMENT_AUDIO_DIR, TEMPLATES_DIR

# app = FastAPI(title="Audio Stream API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["GET", "HEAD", "OPTIONS"],
#     allow_headers=["Range", "Content-Type"],
#     expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"],
# )

console = Console()
router = APIRouter(prefix="/segments", tags=["segments"])

# AUDIO_DIR = Path("audio_files")
# AUDIO_DIR.mkdir(exist_ok=True)

AUDIO_DIR = SEGMENT_AUDIO_DIR

CHUNK_SIZE = 256 * 1024
INITIAL_BUFFER = 512 * 1024


# ---------------------------------------------------------------------------
# Jinja2 template environment
# ---------------------------------------------------------------------------
_templates_dir = TEMPLATES_DIR / "segments"
_templates_dir.mkdir(parents=True, exist_ok=True)

_jinja_env = Environment(
    loader=FileSystemLoader(str(_templates_dir)),
    autoescape=select_autoescape(["html", "xml"]),
)

console.print(f"[info]Segment templates directory: {_templates_dir}[/info]")


def get_template(name: str):
    """Get a Jinja2 template by name with caching."""
    try:
        template = _jinja_env.get_template(name)
        console.print(f"[dim]Loaded template: {name}[/dim]")
        return template
    except Exception as e:
        console.print(f"[error]Failed to load template {name}: {e}[/error]")
        raise HTTPException(
            status_code=500, detail=f"Template {name} not found or invalid"
        )


def render_template(name: str, context: dict = None) -> str:
    """Render a template with context."""
    template = get_template(name)
    return template.render(**(context or {}))


def _get_audio_files() -> dict[str, str]:
    """
    Scan AUDIO_DIR for supported audio files.
    Returns dict mapping base name (stem) → absolute path.
    """
    audio_file_paths = resolve_audio_paths(AUDIO_DIR)
    console.print(f"[info]Indexing audio files from {AUDIO_DIR}[/info]")

    audio_files = {}
    for path_str in audio_file_paths:
        path = Path(path_str)
        audio_files[path.stem] = str(path.resolve())

    console.print(f"[info]Indexed {len(audio_files)} audio files[/info]")
    return audio_files


def get_audio_path(segment_id: str) -> Path:
    """
    Resolve audio path using indexed lookup with direct fallback.
    """
    audio_files = _get_audio_files()

    if segment_id in audio_files:
        file_path = Path(audio_files[segment_id])
        console.print(f"[dim]Found in index: {file_path}[/dim]")
    else:
        safe_name = Path(segment_id).name
        file_path = (AUDIO_DIR / safe_name).resolve()
        console.print(f"[dim]Direct path fallback: {file_path}[/dim]")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {segment_id}")

    return file_path


# ═══════════════════════════════════════════════════════════════
# Player Route — serves the HTML page
# ═══════════════════════════════════════════════════════════════
@router.get("", response_class=HTMLResponse)
async def get_segments_dashboard():
    # """
    # Serve the audio player HTML page.
    # Open http://localhost:8000/ in your browser to use the player.
    # """
    # player_html = Path(__file__).parent / "player.html"
    # if not player_html.exists():
    #     raise HTTPException(status_code=404, detail="player.html not found")
    # return HTMLResponse(content=player_html.read_text(encoding="utf-8"))
    html_content = render_template(
        "dashboard.html",
        {
            # "title": "Speaker Diarization Dashboard",
            # "timestamp": datetime.now().isoformat(),
        },
    )
    console.print("[success]Segments dashboard rendered successfully[/success]")
    return HTMLResponse(content=html_content)


@router.get("/audio/{segment_id}/info")
async def audio_info(segment_id: str):
    """Returns file metadata (size, duration hint, mime type) for player UI."""
    path = get_audio_path(segment_id)
    file_size = path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(path))
    return {
        "filename": segment_id,
        "size_bytes": file_size,
        "mime_type": mime_type or "audio/mpeg",
        "size_mb": round(file_size / 1024 / 1024, 2),
    }


@router.get("/tracks")
async def list_tracks():
    """List all available audio files using indexed lookup."""
    audio_files = _get_audio_files()
    tracks = []
    for stem, abs_path in sorted(audio_files.items()):
        path = Path(abs_path)
        file_size = path.stat().st_size
        mime_type, _ = mimetypes.guess_type(abs_path)
        tracks.append(
            {
                "filename": path.name,
                "stem": stem,
                "size_bytes": file_size,
                "mime_type": mime_type or "audio/mpeg",
            }
        )
    console.print(f"[info]Listed {len(tracks)} tracks[/info]")
    return {"tracks": tracks}


# Add this route after the existing @router.get("/audio/{segment_id}") endpoint:


@router.get("/audio/{segment_id}/download")
async def download_audio(segment_id: str):
    """
    Download endpoint that serves the complete audio file.
    Sets Content-Disposition to force browser download.

    This is different from /audio/{segment_id} which supports Range requests
    for progressive streaming. This endpoint sends the full file.

    Flow:
    1. Resolve and validate the audio file path
    2. Set Content-Disposition header for download
    3. Return FileResponse with appropriate headers
    """
    path = get_audio_path(segment_id)
    file_size = path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "audio/wav"

    # Generate a clean filename for download
    # Remove any problematic characters from segment_id
    safe_filename = f"segment_{segment_id.replace('/', '_').replace('\\', '_')}"
    if path.suffix:
        safe_filename = f"{safe_filename}{path.suffix}"
    else:
        safe_filename = f"{safe_filename}.wav"

    console.print(
        f"[info]Download request for segment {segment_id} -> {safe_filename} ({file_size} bytes)[/]"
    )

    return FileResponse(
        path=str(path),
        media_type=mime_type,
        filename=safe_filename,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@router.get("/segment/{segment_id}", response_class=HTMLResponse)
async def get_segment_page(request: Request, segment_id: str):
    """
    Serve a detailed page for a specific segment with audio player.
    Replicates the speaker's segment detail page but under /segments/segment/{id}

    This is a convenience route that mirrors /speakers/segment/{segment_id}
    but uses the segments router's audio endpoint (/segments/audio/{id}).

    Flow:
    1. Get segment info from speaker labeler
    2. Check audio availability in permanent storage, context buffer, and disk
    3. Render segment_detail.html with appropriate context
    4. Audio player uses /segments/audio/{segment_id} as source
    """
    from services.audio_utils import get_audio_duration

    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(
            status_code=400,
            detail="Speaker labeler not initialized. Process some audio segments first.",
        )

    console.print(
        f"[info]Rendering segment page for: {segment_id} via /segments route[/]"
    )

    if not hasattr(labeler, "get_segment_detail"):
        raise HTTPException(status_code=500, detail="Segment detail not available.")

    segment_info = labeler.get_segment_detail(segment_id)
    if segment_info is None:
        console.print(f"[warning]Segment not found: {segment_id}[/]")
        html_content = render_template(
            "segment_detail.html",
            {
                "title": f"Segment: {segment_id}",
                "segment_id": segment_id,
                "found": False,
                "timestamp": datetime.now().isoformat(),
                "has_audio": False,
                "audio_api_base": "/segments",
            },
        )
        return HTMLResponse(content=html_content, status_code=404)

    # Check audio availability
    has_audio = False
    audio_source = ""
    audio_duration = segment_info.get("segment_duration", 0.0)
    audio_sample_rate = SAMPLE_RATE

    # Check permanent storage
    try:
        if SEGMENT_AUDIO_DIR and SEGMENT_AUDIO_DIR.exists():
            audio_path = SEGMENT_AUDIO_DIR / f"{segment_id}.wav"
            if audio_path.exists():
                has_audio = True
                audio_source = "permanent_storage"
                disk_duration = get_audio_duration(str(audio_path))
                if audio_duration <= 0.0 or disk_duration > 0:
                    audio_duration = disk_duration
                console.print(
                    f"[dim]Audio found in permanent storage: {audio_path} ({disk_duration:.3f}s)[/]"
                )
    except Exception as e:
        console.print(f"[dim]Error checking permanent storage: {e}[/]")

    # Check context buffer
    if not has_audio:
        try:
            from services.context_buffer import get_context_buffer

            context_buffer = get_context_buffer()
            if context_buffer and hasattr(context_buffer, "segments"):
                for segment_audio, metadata in context_buffer.segments:
                    if metadata.get("segment_id") == segment_id:
                        has_audio = True
                        audio_source = "context_buffer"
                        raw_duration = (
                            get_audio_duration(segment_audio, sr=SAMPLE_RATE)
                            if segment_audio is not None
                            else 0.0
                        )
                        if audio_duration <= 0.0 and raw_duration > 0.0:
                            audio_duration = raw_duration
                        break
        except Exception as e:
            console.print(f"[dim]Could not check context buffer: {e}[/]")

    # Check disk fallback
    if not has_audio:
        try:
            from services.config import LAST_N_SEGMENTS_DIR

            last_n_dir = LAST_N_SEGMENTS_DIR
            if last_n_dir and last_n_dir.exists():
                audio_path = last_n_dir / f"{segment_id}.wav"
                if audio_path.exists():
                    has_audio = True
                    audio_source = "disk"
                    disk_duration = get_audio_duration(str(audio_path))
                    if audio_duration <= 0.0:
                        audio_duration = disk_duration
        except Exception:
            pass

    console.print(
        f"[info]Segment: speaker={segment_info['speaker_label']}, "
        f"audio={'yes' if has_audio else 'no'} ({audio_source}), "
        f"duration={audio_duration:.3f}s[/]"
    )

    html_content = render_template(
        "segment_detail.html",
        {
            "title": f"Segment: {segment_id}",
            "segment_id": segment_id,
            "found": True,
            "speaker_label": segment_info["speaker_label"],
            "timestamp": datetime.now().isoformat(),
            "segment_timestamp": segment_info["timestamp"],
            "segment_duration": segment_info["segment_duration"],
            "embedding_index": segment_info["embedding_index"],
            "embedding_dim": segment_info["embedding_dim"],
            "speaker_segment_count": segment_info["speaker_segment_count"],
            "speaker_first_seen": segment_info["speaker_first_seen"],
            "speaker_last_seen": segment_info["speaker_last_seen"],
            "speaker_active_duration": segment_info["speaker_active_duration"],
            "centroid_quality": segment_info["centroid_quality"],
            "has_audio": has_audio,
            "audio_source": audio_source,
            "audio_sample_rate": audio_sample_rate,
            "audio_duration": audio_duration,
            # Point to segments router's audio endpoint
            "audio_api_base": "/segments",
        },
    )

    console.print(f"[success]Segment page rendered for {segment_id}[/]")
    return HTMLResponse(content=html_content)


# Streaming routes


def get_range_info(range_header: Optional[str], file_size: int) -> tuple[int, int]:
    """
    Parse the Range header and return start and end bytes.

    Returns:
        tuple: (start_byte, end_byte) - inclusive range
    """
    if not range_header:
        return 0, file_size - 1

    range_match = re.search(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        console.print(f"[warning]Invalid Range header format: {range_header}[/]")
        return 0, file_size - 1

    start = int(range_match.group(1))
    end_str = range_match.group(2)
    end = int(end_str) if end_str else file_size - 1

    if start >= file_size or end >= file_size or start > end:
        console.print(
            f"[warning]Invalid range: {start}-{end} for file size {file_size}[/]"
        )
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")

    return start, end


def generate_audio_chunks(
    file_path: Path, start: int, end: int, chunk_size: int = 65536
):
    """
    Generator that yields audio file chunks for streaming.
    Uses chunked reading to avoid loading the entire file into memory.
    """
    console.print(
        f"[info]Streaming bytes {start}-{end} with chunk size {chunk_size}[/]"
    )
    with file_path.open("rb") as f:
        f.seek(start)
        bytes_remaining = end - start + 1
        while bytes_remaining > 0:
            read_size = min(chunk_size, bytes_remaining)
            chunk = f.read(read_size)
            if not chunk:
                break
            bytes_remaining -= len(chunk)
            yield chunk
    console.print(f"[info]Finished streaming - {end - start + 1} bytes sent[/]")


@router.get("/audio/{segment_id}")
async def serve_audio(request: Request, segment_id: str):
    """
    Serves the audio file with Range support for progressive streaming.

    Flow:
    1. Check if audio file exists
    2. Get file size
    3. Parse Range header (if present)
    4. Return 206 Partial Content or 200 Full Content
    5. Stream using generator for memory efficiency
    """
    audio_file_path = Path(get_audio_path(segment_id))
    if not audio_file_path.exists():
        console.print(f"[error]Audio file not found at {audio_file_path}[/]")
        raise HTTPException(status_code=404, detail="File not found")

    file_size = audio_file_path.stat().st_size
    range_header = request.headers.get("Range")
    console.print(
        f"[info]GET /audio - Range: {range_header!r} from {request.client.host}[/]"
    )

    start, end = get_range_info(range_header, file_size)
    content_length = end - start + 1

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "audio/mpeg",
    }

    if range_header:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        status_code = 206
        console.print(
            f"[info]206 Partial - bytes {start}-{end}/{file_size} ({content_length} bytes)[/]"
        )
    else:
        status_code = 200
        console.print(f"[info]200 Full - serving entire file ({file_size} bytes)[/]")

    generator = generate_audio_chunks(audio_file_path, start, end)
    return StreamingResponse(
        generator,
        status_code=status_code,
        headers=headers,
        media_type="audio/mpeg",
    )


@router.options("/audio/{segment_id}")
async def audio_options():
    """Handle CORS preflight for audio endpoint."""
    console.print("[info]OPTIONS /audio - preflight request[/]")
    return {"message": "OK"}


@router.get("/health/{segment_id}")
async def health_check(segment_id: str):
    """Health check endpoint for monitoring."""
    audio_file_path = Path(get_audio_path(segment_id))
    exists = audio_file_path.exists()
    console.print(f"[info]GET /health - audio_file_exists={exists}[/]")
    return {"status": "healthy", "audio_file_exists": exists}
