"""
FastAPI Backend for MP4 Chunk Uploads
Endpoints:
- POST /upload-file/ - Accept complete video file as FormData
- POST /upload-chunk/ - Accept raw bytes chunks
- GET /health - Health check
"""

import os
import logging
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MP4 Chunk Uploader",
    description="Receives video/audio chunks from Chrome extension",
    version="1.0.0"
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Store chunk sessions for Option B (optional - for reassembly)
chunk_sessions = {}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy", "upload_dir": str(UPLOAD_DIR)}


@app.post("/upload-file/")
async def upload_complete_file(
    file: UploadFile = File(..., description="Video or audio file to upload")
):
    """
    Option A: Upload complete file using FormData
    Accepts video/mp4, audio/webm, etc.
    """
    logger.info(f"=== Option A: Complete File Upload ===")
    logger.info(f"Filename: {file.filename}")
    logger.info(f"Content Type: {file.content_type}")
    
    # Validate file type
    if not file.content_type or not (
        file.content_type.startswith("video/") or 
        file.content_type.startswith("audio/")
    ):
        logger.warning(f"Invalid content type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Expected video/* or audio/*, got {file.content_type}"
        )
    
    try:
        # Generate unique filename
        original_name = Path(file.filename).stem
        extension = Path(file.filename).suffix or ".mp4"
        unique_filename = f"{original_name}_{uuid.uuid4().hex[:8]}{extension}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save file efficiently using streaming
        file_size = 0
        with file_path.open("wb") as buffer:
            # Copy file contents in chunks (handles large files efficiently)
            while chunk := await file.read(8192):  # 8KB chunks
                file_size += len(chunk)
                buffer.write(chunk)
        
        logger.info(f"✅ File saved successfully: {file_path} ({file_size:,} bytes)")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "filename": unique_filename,
                "original_filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "path": str(file_path)
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Failed to save file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )


@app.post("/upload-chunk/")
async def upload_chunk_bytes(
    chunk: bytes = File(..., description="Raw audio/video chunk bytes"),
    chunk_index: Optional[int] = None,
    session_id: Optional[str] = None,
    filename: Optional[str] = None
):
    """
    Option B: Upload raw bytes chunks for streaming
    Query params:
    - chunk_index: Index of chunk in sequence (0, 1, 2...)
    - session_id: Unique ID for this upload session
    - filename: Optional output filename
    """
    logger.info(f"=== Option B: Chunk Bytes Upload ===")
    logger.info(f"Chunk size: {len(chunk):,} bytes")
    logger.info(f"Chunk index: {chunk_index}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Filename: {filename}")
    
    # Validate chunk size
    if len(chunk) == 0:
        logger.warning("Received empty chunk")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty chunk received"
        )
    
    # Generate session ID if not provided
    if not session_id:
        session_id = uuid.uuid4().hex
        logger.info(f"Generated new session ID: {session_id}")
    
    # Initialize session if new
    if session_id not in chunk_sessions:
        chunk_sessions[session_id] = {
            "chunks": [],
            "total_bytes": 0,
            "filename": filename or f"recording_{session_id}.mp4",
            "chunk_count": 0
        }
    
    session = chunk_sessions[session_id]
    session["chunks"].append({
        "index": chunk_index if chunk_index is not None else len(session["chunks"]),
        "size": len(chunk),
        "data": chunk  # Store for processing (or write to disk)
    })
    session["total_bytes"] += len(chunk)
    session["chunk_count"] += 1
    
    logger.info(f"Session {session_id}: Chunk {len(session['chunks'])} received. "
                f"Total: {session['total_bytes']:,} bytes")
    
    # Optional: Write chunks to disk incrementally
    chunk_file_path = UPLOAD_DIR / f"{session_id}_chunk_{len(session['chunks']):04d}.bin"
    with chunk_file_path.open("wb") as f:
        f.write(chunk)
    logger.debug(f"Chunk saved to: {chunk_file_path}")
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "session_id": session_id,
            "chunk_index": len(session["chunks"]),
            "chunk_size_bytes": len(chunk),
            "session_total_bytes": session["total_bytes"],
            "session_chunk_count": session["chunk_count"]
        }
    )


@app.post("/upload-chunk/finalize/")
async def finalize_chunk_session(session_id: str):
    """
    Finalize a chunk session and reassemble the complete file
    """
    logger.info(f"=== Finalizing session: {session_id} ===")
    
    if session_id not in chunk_sessions:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    session = chunk_sessions[session_id]
    
    # Sort chunks by index
    session["chunks"].sort(key=lambda x: x["index"])
    
    # Reassemble file
    output_filename = session["filename"]
    output_path = UPLOAD_DIR / output_filename
    
    with output_path.open("wb") as outfile:
        for chunk_info in session["chunks"]:
            outfile.write(chunk_info["data"])
    
    logger.info(f"✅ Session finalized: {output_path} ({session['total_bytes']:,} bytes)")
    
    # Clean up individual chunk files (optional)
    for i in range(1, session["chunk_count"] + 1):
        chunk_file = UPLOAD_DIR / f"{session_id}_chunk_{i:04d}.bin"
        if chunk_file.exists():
            chunk_file.unlink()
            logger.debug(f"Removed chunk file: {chunk_file}")
    
    # Remove session data
    final_size = session["total_bytes"]
    del chunk_sessions[session_id]
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "session_id": session_id,
            "final_filename": output_filename,
            "final_size_bytes": final_size,
            "final_size_mb": round(final_size / (1024 * 1024), 2),
            "path": str(output_path)
        }
    )


@app.get("/sessions/")
async def list_sessions():
    """List all active chunk sessions"""
    sessions_info = {}
    for session_id, session in chunk_sessions.items():
        sessions_info[session_id] = {
            "chunk_count": session["chunk_count"],
            "total_bytes": session["total_bytes"],
            "filename": session["filename"]
        }
    return {"active_sessions": sessions_info}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
