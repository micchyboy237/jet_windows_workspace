"""
Audio tagging routes for sound event detection and analysis.
"""
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from rich.console import Console
from rich.table import Table
from jinja2 import Template

from services.audio_tagger import AudioTagger, TaggingResult, AudioChunksTaggingSummary
from core.state import (
    get_audio_tagger,
    set_audio_tagger,
    get_last_n_segments_dir,
)
from services.save_utils import (
    save_tagging_results,
    save_chunked_results,
)
from config import OUTPUT_DIR, TEMPLATES_DIR

templates_dir = TEMPLATES_DIR / "tagger"

console = Console()
router = APIRouter(prefix="/tags", tags=["audio-tagging"])


def get_tagger() -> AudioTagger:
    """
    Get or initialize the audio tagger singleton.
    
    Uses core.state for centralized state management,
    consistent with other singletons in the application.
    """
    tagger = get_audio_tagger()
    if tagger is None:
        console.print("[info]Initializing AudioTagger...[/info]")
        tagger = AudioTagger(
            top_k=5,
            speech_prob_threshold=0.5,
            chunk_duration=2.0,
            chunk_overlap=1.0,
            debug=False,
        )
        set_audio_tagger(tagger)
        console.print("[success]AudioTagger initialized successfully[/success]")
    return tagger


@router.get("", response_class=HTMLResponse)
async def get_tags():
    """
    Display audio tagging data visualizations using Chart.js.
    Shows:
    - Summary statistics at the top
    - Speech probability distribution over segments
    - Top predictions frequency bar chart
    - Speech vs non-speech timeline
    - Processing mode distribution
    - Speech probability trend line
    """
    template_path = templates_dir / "tags.html"
    template = Template(template_path.read_text(encoding='utf-8'))
    return HTMLResponse(content=template.render())


@router.get("/config")
async def get_tagger_config():
    """Get current audio tagger configuration."""
    tagger = get_tagger()
    
    return {
        "model_path": str(tagger.model_path),
        "labels_path": str(tagger.labels_path),
        "top_k": tagger.top_k,
        "speech_prob_threshold": tagger.speech_prob_threshold,
        "speech_top_n": tagger.speech_top_n,
        "chunk_duration": tagger.chunk_duration,
        "chunk_overlap": tagger.chunk_overlap,
        "min_chunk_duration": tagger.min_chunk_duration,
        "speech_classes": tagger.SPEECH_CLASS_NAMES,
    }


@router.get("/chunks", response_class=JSONResponse)
async def get_saved_chunks(
    limit: Optional[int] = Query(100, description="Maximum number of chunks to return", ge=1, le=500),
    offset: Optional[int] = Query(0, description="Number of chunks to skip", ge=0),
    speech_only: Optional[bool] = Query(False, description="Filter to only show segments with speech detected"),
):
    """
    Get saved audio tagging chunk data from the global all_tag_events.json file.
    
    This endpoint reads the accumulated tag events stored in the last_n_segments_dir
    and returns them with optional filtering and pagination.
    
    Parameters:
    - limit: Maximum number of entries to return (1-500, default 100)
    - offset: Number of entries to skip for pagination (default 0)
    - speech_only: If true, only return segments where speech was detected
    
    Returns:
    - total_entries: Total number of entries in the file
    - returned_entries: Number of entries in this response
    - offset: Current offset
    - limit: Current limit
    - chunks: Array of tag event entries
    - stats: Summary statistics about the tag events
    """
    console.print(f"[info]📊 Fetching saved chunks data[/info]")
    console.print(f"[dim]Parameters: limit={limit}, offset={offset}, speech_only={speech_only}[/dim]")
    
    last_n_segments_dir = get_last_n_segments_dir()
    tag_events_path = last_n_segments_dir / "all_tag_events.json"
    
    console.print(f"[dim]Reading from: {tag_events_path}[/dim]")
    
    if not tag_events_path.exists():
        console.print(f"[warning]No tag events file found at {tag_events_path}[/warning]")
        return JSONResponse(content={
            "success": True,
            "total_entries": 0,
            "returned_entries": 0,
            "offset": offset,
            "limit": limit,
            "chunks": [],
            "stats": {
                "total_segments": 0,
                "speech_segments": 0,
                "non_speech_segments": 0,
                "speech_percentage": 0.0,
                "avg_speech_probability": 0.0,
                "processing_modes": {},
                "top_predictions": [],
                "timestamp": datetime.now().isoformat(),
            },
            "message": "No tag events recorded yet. Process some audio segments first.",
        })
    
    try:
        # Read the tag events file
        with open(tag_events_path, 'r', encoding='utf-8') as f:
            all_entries = json.load(f)
        
        console.print(f"[dim]Read {len(all_entries)} total entries from file[/dim]")
        
        if not isinstance(all_entries, list):
            console.print(f"[warning]Tag events file is not a list, resetting[/warning]")
            all_entries = []
        
        # Calculate statistics on all entries before filtering
        total_segments = len(all_entries)
        speech_segments = sum(1 for entry in all_entries if entry.get("speech_detected", False))
        non_speech_segments = total_segments - speech_segments
        
        # Calculate average speech probability
        speech_probs = [entry.get("speech_probability", 0.0) for entry in all_entries if entry.get("speech_probability") is not None]
        avg_speech_prob = sum(speech_probs) / len(speech_probs) if speech_probs else 0.0
        
        # Count processing modes
        processing_modes = {}
        for entry in all_entries:
            mode = entry.get("processing_mode", "unknown")
            processing_modes[mode] = processing_modes.get(mode, 0) + 1
        
        # Aggregate top predictions across all entries
        prediction_counts = {}
        for entry in all_entries:
            for pred in entry.get("top_predictions", []):
                name = pred.get("name", "Unknown")
                if name not in prediction_counts:
                    prediction_counts[name] = {
                        "count": 0,
                        "total_prob": 0.0,
                    }
                prediction_counts[name]["count"] += 1
                prediction_counts[name]["total_prob"] += pred.get("prob", 0.0)
        
        # Sort predictions by frequency
        sorted_predictions = sorted(
            prediction_counts.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        top_predictions = [
            {
                "name": name,
                "count": stats["count"],
                "avg_probability": round(stats["total_prob"] / stats["count"], 4),
            }
            for name, stats in sorted_predictions[:10]
        ]
        
        # Apply filters
        filtered_entries = all_entries
        if speech_only:
            filtered_entries = [entry for entry in all_entries if entry.get("speech_detected", False)]
            console.print(f"[dim]Filtered to {len(filtered_entries)} speech-only entries[/dim]")
        
        # Apply pagination
        paginated_entries = filtered_entries[offset:offset + limit]
        
        console.print(f"[dim]Returning {len(paginated_entries)} entries (offset={offset}, limit={limit})[/dim]")
        
        response = {
            "success": True,
            "total_entries": len(filtered_entries),
            "returned_entries": len(paginated_entries),
            "offset": offset,
            "limit": limit,
            "chunks": paginated_entries,
            "stats": {
                "total_segments": total_segments,
                "speech_segments": speech_segments,
                "non_speech_segments": non_speech_segments,
                "speech_percentage": round(speech_segments / max(total_segments, 1) * 100, 1),
                "avg_speech_probability": round(avg_speech_prob, 4),
                "processing_modes": processing_modes,
                "top_predictions": top_predictions,
                "timestamp": datetime.now().isoformat(),
            },
        }
        
        console.print(f"[success]✅ Returned {len(paginated_entries)} chunks with statistics[/success]")
        return JSONResponse(content=response)
        
    except json.JSONDecodeError as e:
        console.print(f"[error]Failed to parse tag events file: {e}[/error]")
        raise HTTPException(status_code=500, detail=f"Failed to parse tag events data: {str(e)}")
    except Exception as e:
        console.print(f"[error]Error reading tag events: {e}[/error]")
        console.print(f"[dim]Traceback: {traceback.format_exc()}[/dim]")
        raise HTTPException(status_code=500, detail=f"Error reading tag events: {str(e)}")


@router.get("/dashboard", response_class=HTMLResponse)
async def get_tagger_dashboard():
    tagger_config = await get_tagger_config()
    template_path = templates_dir / "dashboard.html"
    template = Template(template_path.read_text(encoding='utf-8'))
    return HTMLResponse(content=template.render(config=tagger_config))


@router.post("/audio")
async def tag_audio_endpoint(
    file: UploadFile = File(..., description="Audio file to tag (WAV, MP3, etc.)"),
    sample_rate: int = Form(16000, description="Sample rate for processing"),
    top_k: Optional[int] = Form(5, description="Number of top predictions to return"),
    chunked: bool = Form(False, description="Process audio in chunks"),
    chunk_duration: Optional[float] = Form(None, description="Chunk duration in seconds (for chunked mode)"),
    overlap_duration: Optional[float] = Form(None, description="Overlap duration in seconds (for chunked mode)"),
    min_chunk_duration: Optional[float] = Form(None, description="Minimum chunk duration in seconds"),
):
    """
    Tag audio file for sound events and speech detection.
    
    Supports both regular and chunked processing modes:
    - Regular: Tags entire audio file at once (best for files < 30s)
    - Chunked: Splits audio into overlapping chunks (best for long files)
    
    Returns predictions, speech detection status, and processing metadata.
    """
    tagger = get_tagger()
    
    try:
        audio_bytes = await file.read()
        
        start_time = time.time()
        
        if chunked:
            # Process audio in chunks
            summary: AudioChunksTaggingSummary = tagger.tag_audio_chunks(
                audio=audio_bytes,
                sample_rate=sample_rate,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                min_chunk_duration=min_chunk_duration,
            )
            
            elapsed = time.time() - start_time
            
            response = {
                "success": True,
                "mode": "chunked",
                "filename": file.filename,
                "file_size_bytes": len(audio_bytes),
                "total_duration_seconds": summary["total_duration"],
                "sample_rate": summary["sample_rate"],
                "chunk_duration": summary["chunk_duration"],
                "overlap_duration": summary["overlap_duration"],
                "total_chunks": summary["total_chunks"],
                "overall_top_predictions": summary["overall_top_predictions"],
                "chunks": [
                    {
                        "chunk_index": chunk["chunk_index"],
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                        "duration": chunk["duration"],
                        "top_predictions": chunk["predictions"][:3],  # Top 3 per chunk
                    }
                    for chunk in summary["chunks"]
                ],
                "speech_detected": any(
                    any(
                        "speech" in pred["name"].lower() and pred["prob"] >= 0.5
                        for pred in chunk["predictions"]
                    )
                    for chunk in summary["chunks"]
                ),
                "processing_time_seconds": round(elapsed, 4),
                "real_time_factor": summary["real_time_factor"],
                "timestamp": datetime.now().isoformat(),
            }
        else:
            # Process entire audio at once
            results: List[TaggingResult] = tagger.tag_audio(audio_bytes, sample_rate=sample_rate)
            
            elapsed = time.time() - start_time
            
            # Check for speech
            speech_prob = tagger.get_speech_probability(audio_bytes, sample_rate=sample_rate)
            
            response = {
                "success": True,
                "mode": "full",
                "filename": file.filename,
                "file_size_bytes": len(audio_bytes),
                "sample_rate": sample_rate,
                "num_predictions": len(results),
                "top_predictions": results[:top_k],
                "speech_detected": speech_prob >= 0.5,
                "max_speech_probability": round(speech_prob, 4),
                "processing_time_seconds": round(elapsed, 4),
                "real_time_factor": round(elapsed / (len(audio_bytes) / (2 * sample_rate)), 4) if sample_rate > 0 else 0,
                "timestamp": datetime.now().isoformat(),
            }
        
        # Save results if output directory exists
        save_results = save_tagging_results(file.filename, response, OUTPUT_DIR)
        if save_results:
            response["saved_to"] = str(save_results)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        console.print(f"[error]Audio tagging failed: {e}[/error]")
        raise HTTPException(status_code=500, detail=f"Tagging failed: {str(e)}")


@router.post("/chunks")
async def tag_audio_chunks_endpoint(
    file: UploadFile = File(..., description="Audio file to tag in chunks"),
    sample_rate: int = Form(16000, description="Sample rate for processing"),
    chunk_duration: float = Form(2.0, description="Duration of each chunk in seconds"),
    overlap_duration: float = Form(1.0, description="Overlap between chunks in seconds"),
    min_chunk_duration: Optional[float] = Form(None, description="Minimum chunk duration"),
    top_k: int = Form(5, description="Number of top predictions to return"),
):
    """
    Tag audio file in overlapping chunks for temporal analysis.
    
    Useful for:
    - Long recordings where audio content changes over time
    - Tracking speech/music patterns across a recording
    - Finding specific sound events with timestamps
    """
    tagger = get_tagger()
    
    try:
        audio_bytes = await file.read()
        
        start_time = time.time()
        
        summary: AudioChunksTaggingSummary = tagger.tag_audio_chunks(
            audio=audio_bytes,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            min_chunk_duration=min_chunk_duration,
        )
        
        elapsed = time.time() - start_time
        
        # Build detailed response
        chunks_data = []
        for chunk in summary["chunks"]:
            chunks_data.append({
                "chunk_index": chunk["chunk_index"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "duration": chunk["duration"],
                "predictions": chunk["predictions"][:top_k],
                "processing_time": chunk["processing_time"],
                "has_speech": any(
                    pred["name"] in tagger.SPEECH_CLASS_NAMES and pred["prob"] >= 0.5
                    for pred in chunk["predictions"]
                ),
            })
        
        response = {
            "success": True,
            "filename": file.filename,
            "file_size_bytes": len(audio_bytes),
            "total_duration_seconds": summary["total_duration"],
            "sample_rate": summary["sample_rate"],
            "chunk_duration": summary["chunk_duration"],
            "overlap_duration": summary["overlap_duration"],
            "total_chunks": summary["total_chunks"],
            "chunks": chunks_data,
            "overall_top_predictions": summary["overall_top_predictions"],
            "speech_segments_count": sum(1 for c in chunks_data if c["has_speech"]),
            "speech_coverage_percentage": round(
                sum(1 for c in chunks_data if c["has_speech"]) / max(len(chunks_data), 1) * 100, 1
            ),
            "processing_time_seconds": round(elapsed, 4),
            "real_time_factor": summary["real_time_factor"],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results
        save_results = save_chunked_results(file.filename, response, OUTPUT_DIR)
        if save_results:
            response["saved_to"] = str(save_results)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        console.print(f"[error]Audio chunk tagging failed: {e}[/error]")
        raise HTTPException(status_code=500, detail=f"Chunk tagging failed: {str(e)}")


@router.post("/speech-check")
async def check_speech_endpoint(
    file: UploadFile = File(..., description="Audio file to check for speech"),
    sample_rate: int = Form(16000, description="Sample rate for processing"),
    threshold: float = Form(0.5, description="Speech probability threshold (0.0-1.0)"),
):
    """
    Quick check if audio contains speech.
    
    Returns speech detection result with probability score.
    """
    tagger = get_tagger()
    
    try:
        audio_bytes = await file.read()
        
        start_time = time.time()
        
        has_speech = tagger.contains_speech(audio_bytes, sample_rate=sample_rate, prob_threshold=threshold)
        speech_prob = tagger.get_speech_probability(audio_bytes, sample_rate=sample_rate)
        
        elapsed = time.time() - start_time
        
        response = {
            "success": True,
            "filename": file.filename,
            "has_speech": has_speech,
            "speech_probability": round(speech_prob, 4),
            "threshold_used": threshold,
            "processing_time_seconds": round(elapsed, 4),
            "timestamp": datetime.now().isoformat(),
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        console.print(f"[error]Speech check failed: {e}[/error]")
        raise HTTPException(status_code=500, detail=f"Speech check failed: {str(e)}")


@router.post("/config/update")
async def update_tagger_config(
    top_k: Optional[int] = Form(None),
    speech_prob_threshold: Optional[float] = Form(None),
    chunk_duration: Optional[float] = Form(None),
    chunk_overlap: Optional[float] = Form(None),
):
    """Update audio tagger configuration."""
    global _tagger_instance
    
    if _tagger_instance is None:
        raise HTTPException(status_code=400, detail="Tagger not initialized")
    
    tagger = _tagger_instance
    
    if top_k is not None:
        tagger.top_k = top_k
    if speech_prob_threshold is not None:
        tagger.speech_prob_threshold = speech_prob_threshold
    if chunk_duration is not None:
        tagger.chunk_duration = chunk_duration
    if chunk_overlap is not None:
        tagger.chunk_overlap = chunk_overlap
    
    tagger._validate_chunking_config()
    
    console.print("[success]AudioTagger configuration updated[/success]")
    
    return {
        "success": True,
        "message": "Configuration updated",
        "current_config": await get_tagger_config(),
    }
