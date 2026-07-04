import logging
import os
import re
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse

BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = str(Path("~/.cache/files/audio").expanduser().resolve())

logger = logging.getLogger(__name__)
router = APIRouter()

# Supported audio extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a', '.wma'}

def get_audio_files() -> list[dict]:
    """
    Recursively collect all audio files from AUDIO_DIR.
    Returns list of dicts with relative path, display name, and full path.
    """
    audio_files = []
    if not os.path.exists(AUDIO_DIR):
        logger.warning(f"Audio directory does not exist: {AUDIO_DIR}")
        return audio_files
    
    for root, dirs, files in os.walk(AUDIO_DIR):
        for file in files:
            if Path(file).suffix.lower() in AUDIO_EXTENSIONS:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(AUDIO_DIR)
                # Create a unique name from relative path (without extension)
                unique_name = str(rel_path.with_suffix('')).replace(os.sep, ' / ')
                audio_files.append({
                    'name': unique_name,
                    'path': str(rel_path),
                    'full_path': str(full_path),
                    'size': os.path.getsize(full_path),
                    'extension': full_path.suffix.lower()
                })
    
    logger.info(f"Found {len(audio_files)} audio files in {AUDIO_DIR}")
    return sorted(audio_files, key=lambda x: x['name'])

def resolve_audio_path(file_param: str) -> str:
    """
    Resolve and validate the audio file path.
    Prevents directory traversal attacks.
    """
    requested_path = Path(file_param)
    safe_path = Path(*[p for p in requested_path.parts if p != '..'])
    full_path = Path(AUDIO_DIR) / safe_path
    
    try:
        full_path.resolve().relative_to(Path(AUDIO_DIR).resolve())
    except ValueError:
        logger.error(f"Path traversal attempt: {file_param}")
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not full_path.exists():
        logger.error(f"Audio file not found: {full_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    return str(full_path)

def get_range_info(range_header: Optional[str], file_size: int) -> tuple[int, int]:
    """Parse the Range header and return start and end bytes."""
    if not range_header:
        return 0, file_size - 1
    
    range_match = re.search(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        logger.warning(f"Invalid Range header format: {range_header}")
        return 0, file_size - 1
    
    start = int(range_match.group(1))
    end_str = range_match.group(2)
    end = int(end_str) if end_str else file_size - 1
    
    if start >= file_size or end >= file_size or start > end:
        logger.warning(f"Invalid range: {start}-{end} for file size {file_size}")
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")
    
    return start, end

def generate_audio_chunks(file_path: str, start: int, end: int, chunk_size: int = 65536):
    """Generator that yields audio file chunks for streaming."""
    logger.info(f"Streaming bytes {start}-{end} with chunk size {chunk_size}")
    with open(file_path, "rb") as f:
        f.seek(start)
        bytes_remaining = end - start + 1
        while bytes_remaining > 0:
            read_size = min(chunk_size, bytes_remaining)
            chunk = f.read(read_size)
            if not chunk:
                break
            bytes_remaining -= len(chunk)
            yield chunk
    logger.info(f"Finished streaming - {end - start + 1} bytes sent")

@router.get("/audio-list")
async def list_audio_files():
    """Returns a JSON list of available audio files."""
    logger.info("GET /audio-list - fetching available audio files")
    audio_files = get_audio_files()
    return JSONResponse(content={
        'audio_dir': AUDIO_DIR,
        'count': len(audio_files),
        'files': audio_files
    })

@router.get("/audio")
async def serve_audio(request: Request, file: str = Query(None, description="Relative path to audio file")):
    """Serves the audio file with Range support for progressive streaming."""
    if file:
        audio_file_path = resolve_audio_path(file)
    else:
        audio_files = get_audio_files()
        if not audio_files:
            raise HTTPException(status_code=404, detail="No audio files available")
        audio_file_path = audio_files[0]['full_path']
        logger.info(f"No file specified, using default: {audio_files[0]['name']}")
    
    file_size = os.path.getsize(audio_file_path)
    range_header = request.headers.get("Range")
    logger.info(f"GET /audio - File: {Path(audio_file_path).name} - Range: {range_header!r} from {request.client.host}")
    
    start, end = get_range_info(range_header, file_size)
    content_length = end - start + 1
    
    ext = Path(audio_file_path).suffix.lower()
    content_type_map = {
        '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg',
        '.flac': 'audio/flac', '.aac': 'audio/aac', '.m4a': 'audio/mp4',
        '.wma': 'audio/x-ms-wma'
    }
    content_type = content_type_map.get(ext, 'audio/mpeg')
    
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": content_type,
    }
    
    if range_header:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        status_code = 206
        logger.info(f"206 Partial - bytes {start}-{end}/{file_size} ({content_length} bytes)")
    else:
        status_code = 200
        logger.info(f"200 Full - serving entire file ({file_size} bytes)")
    
    generator = generate_audio_chunks(audio_file_path, start, end)
    return StreamingResponse(
        generator,
        status_code=status_code,
        headers=headers,
        media_type=content_type,
    )

@router.options("/audio")
async def audio_options():
    """Handle CORS preflight for audio endpoint."""
    logger.info("OPTIONS /audio - preflight request")
    return {"message": "OK"}

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    audio_dir_exists = os.path.exists(AUDIO_DIR)
    audio_files = get_audio_files() if audio_dir_exists else []
    logger.info(f"GET /health - audio_dir_exists={audio_dir_exists}, files_count={len(audio_files)}")
    return {
        "status": "healthy",
        "audio_dir_exists": audio_dir_exists,
        "audio_files_count": len(audio_files)
    }
