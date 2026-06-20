import logging
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

BASE_DIR = Path(__file__).parent.parent
audio_file_path = str(BASE_DIR / "sample_audio.wav")

logger = logging.getLogger("audio_streaming.audio_router")

router = APIRouter()


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
    """
    Generator that yields audio file chunks for streaming.
    Uses chunked reading to avoid loading the entire file into memory.
    """
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


@router.get("/audio")
async def serve_audio(request: Request):
    """
    Serves the audio file with Range support for progressive streaming.

    Flow:
    1. Check if audio file exists
    2. Get file size
    3. Parse Range header (if present)
    4. Return 206 Partial Content or 200 Full Content
    5. Stream using generator for memory efficiency
    """
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found at {audio_file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(audio_file_path)
    range_header = request.headers.get("Range")
    logger.info(f"GET /audio - Range: {range_header!r} from {request.client.host}")

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
        logger.info(f"206 Partial - bytes {start}-{end}/{file_size} ({content_length} bytes)")
    else:
        status_code = 200
        logger.info(f"200 Full - serving entire file ({file_size} bytes)")

    generator = generate_audio_chunks(audio_file_path, start, end)
    return StreamingResponse(
        generator,
        status_code=status_code,
        headers=headers,
        media_type="audio/mpeg",
    )


@router.options("/audio")
async def audio_options():
    """Handle CORS preflight for audio endpoint."""
    logger.info("OPTIONS /audio - preflight request")
    return {"message": "OK"}


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    exists = os.path.exists(audio_file_path)
    logger.info(f"GET /health - audio_file_exists={exists}")
    return {"status": "healthy", "audio_file_exists": exists}
