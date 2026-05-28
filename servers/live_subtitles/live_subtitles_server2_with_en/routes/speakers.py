"""
Speaker management routes.
"""
from typing import Dict
from fastapi import APIRouter, Form, HTTPException
from core.state import (
    get_speaker_labeler,
    set_current_speaker,
    set_last_speaker_change_time,
    get_context_buffer,
    save_speaker_state,
    get_speaker_state_path,
)
from core.processing import get_speaker_diarization
from rich.console import Console

console = Console()
router = APIRouter(prefix="/speakers", tags=["speakers"])


@router.get("")
async def get_speakers():
    """Get current speaker diarization information."""
    return get_speaker_diarization()


@router.get("/status")
def get_status() -> Dict:
    """Get current speakers status."""
    labeler = get_speaker_labeler()
    if not labeler:
        return {"status": "not_initialized"}
    health_status = labeler.get_health_status()
    return dict(health_status)


@router.get("/similarities")
def get_speaker_similarity_matrix() -> Dict:
    """Get current speakers similarity matrix."""
    labeler = get_speaker_labeler()
    if not labeler:
        return {"error": "Speaker labeler not initialized"}
    speaker_similarity_matrix = labeler.get_speaker_similarity_matrix()
    return dict(speaker_similarity_matrix)


@router.post("/consolidate")
async def consolidate_speakers_endpoint(
    threshold: float = Form(0.85),
    dry_run: bool = Form(False),
):
    """Consolidate similar speakers by merging those above similarity threshold.
    
    Parameters
    ----------
    threshold : float
        Similarity threshold above which speakers are merged (0.0 to 1.0).
    dry_run : bool
        If true, returns proposed merges without executing them.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    result = labeler.consolidate_speakers(threshold=threshold, dry_run=dry_run)
    if not dry_run:
        save_speaker_state()
    
    return {
        "success": True,
        **result,
    }


@router.post("/reset")
async def reset_speakers():
    """Reset speaker labeler state - fully clears all speaker tracking."""
    labeler = get_speaker_labeler()
    if labeler:
        labeler.reset()
    
    set_current_speaker(None)
    set_last_speaker_change_time(0.0)
    
    context_buffer = get_context_buffer()
    if context_buffer.segments:
        for segment_audio, metadata in context_buffer.segments:
            metadata["speaker_label"] = None
            metadata["speaker_confidence"] = 0.0
            metadata["speakers"] = []
    
    speaker_state_path = get_speaker_state_path()
    if speaker_state_path.exists():
        speaker_state_path.unlink()
    
    save_speaker_state()
    console.print("[warning]🔄 Speaker state fully reset: labeler + global state + context buffer[/warning]")
    
    return {"success": True, "message": "Speaker state reset"}


@router.post("/merge")
async def merge_speakers(label1: str = Form(...), label2: str = Form(...)):
    """Merge two speaker labels into one."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    result = labeler.merge_speakers(label1, label2)
    if result is None:
        raise HTTPException(status_code=400, detail=f"Could not merge {label1} and {label2}")
    
    save_speaker_state()
    return {"success": True, "merged_label": result}
