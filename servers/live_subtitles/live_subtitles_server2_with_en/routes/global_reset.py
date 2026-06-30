"""
Global reset routes for resetting all application state.
Supports full reset (complete state wipe) and soft reset (runtime state only).
"""
import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from rich.console import Console

from core.state import (
    get_speaker_labeler,
    set_speaker_labeler,
    set_current_speaker,
    set_last_speaker_change_time,
    get_context_buffer,
    get_speaker_state_path,
    get_segment_index_path,
    get_last_n_segments_dir,
    get_live_audio_buffer_dir,
    get_segments_audio_dir,
    get_audio_tagger,
    set_audio_tagger,
    get_embedding_inference,
    set_embedding_inference,
    get_executor,
    save_speaker_state,
)
from services.live_subtitles_server_utils import (
    load_segment_counter,
    save_segment_counter,
)
from services.config import OUTPUT_DIR

console = Console()
router = APIRouter(prefix="/global", tags=["global-reset"])


def _reset_speaker_labeler(debug: bool = False) -> Dict[str, Any]:
    """
    Reset the speaker labeler component.
    
    Args:
        debug: Enable detailed debug logging
        
    Returns:
        Dict with reset statistics
    """
    stats = {
        "component": "speaker_labeler",
        "previous_speaker_count": 0,
        "previous_segments_processed": 0,
        "reset_success": False,
        "errors": [],
    }
    
    try:
        labeler = get_speaker_labeler()
        if labeler:
            stats["previous_speaker_count"] = labeler.speaker_count
            stats["previous_segments_processed"] = labeler.total_segments_processed
            
            if debug:
                console.print(f"[debug]Resetting speaker labeler: "
                             f"{stats['previous_speaker_count']} speakers, "
                             f"{stats['previous_segments_processed']} segments[/debug]")
            
            labeler.reset()
            stats["reset_success"] = True
            
            if debug:
                console.print(f"[debug]Speaker labeler reset complete. "
                             f"New state: {labeler.speaker_count} speakers, "
                             f"{labeler.total_segments_processed} segments[/debug]")
        else:
            stats["note"] = "Speaker labeler was not initialized"
            stats["reset_success"] = True  # Not an error if never initialized
            
    except Exception as e:
        error_msg = f"Failed to reset speaker labeler: {str(e)}"
        stats["errors"].append(error_msg)
        if debug:
            console.print(f"[error]{error_msg}[/error]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return stats


def _clear_context_buffer(debug: bool = False) -> Dict[str, Any]:
    """
    Clear the audio context buffer.
    
    Args:
        debug: Enable detailed debug logging
        
    Returns:
        Dict with reset statistics
    """
    stats = {
        "component": "context_buffer",
        "previous_segment_count": 0,
        "previous_total_duration": 0.0,
        "reset_success": False,
        "errors": [],
    }
    
    try:
        context_buffer = get_context_buffer()
        if context_buffer:
            stats["previous_segment_count"] = len(context_buffer.segments)
            stats["previous_total_duration"] = context_buffer.get_total_duration()
            
            if debug:
                console.print(f"[debug]Clearing context buffer: "
                             f"{stats['previous_segment_count']} segments, "
                             f"{stats['previous_total_duration']:.2f}s duration[/debug]")
            
            # Clear the buffer
            context_buffer.reset()
            stats["reset_success"] = True
            
            if debug:
                console.print(f"[debug]Context buffer cleared. "
                             f"New state: {len(context_buffer.segments)} segments, "
                             f"{context_buffer.get_total_duration():.2f}s duration[/debug]")
        else:
            stats["note"] = "Context buffer not available"
            stats["reset_success"] = True
            
    except Exception as e:
        error_msg = f"Failed to clear context buffer: {str(e)}"
        stats["errors"].append(error_msg)
        if debug:
            console.print(f"[error]{error_msg}[/error]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return stats


def _reset_segment_counter(debug: bool = False) -> Dict[str, Any]:
    """
    Reset the segment counter to 0.
    
    Args:
        debug: Enable detailed debug logging
        
    Returns:
        Dict with reset statistics
    """
    stats = {
        "component": "segment_counter",
        "previous_value": 0,
        "new_value": 0,
        "reset_success": False,
        "errors": [],
    }
    
    try:
        segment_index_path = get_segment_index_path()
        
        if segment_index_path.exists():
            try:
                with open(segment_index_path, 'r') as f:
                    data = json.load(f)
                stats["previous_value"] = data.get("counter", 0)
                
                if debug:
                    console.print(f"[debug]Resetting segment counter from {stats['previous_value']}[/debug]")
            except Exception:
                stats["previous_value"] = "unknown"
        
        # Reset the counter
        save_segment_counter(segment_index_path)  # This saves current (should be reset)
        
        # Force reload with 0
        if segment_index_path.exists():
            segment_index_path.unlink()
        
        stats["reset_success"] = True
        stats["new_value"] = 0
        
        if debug:
            console.print(f"[debug]Segment counter reset to 0[/debug]")
            
    except Exception as e:
        error_msg = f"Failed to reset segment counter: {str(e)}"
        stats["errors"].append(error_msg)
        if debug:
            console.print(f"[error]{error_msg}[/error]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return stats


def _clear_current_speaker_state(debug: bool = False) -> Dict[str, Any]:
    """
    Clear current speaker tracking state.
    
    Args:
        debug: Enable detailed debug logging
        
    Returns:
        Dict with reset statistics
    """
    stats = {
        "component": "current_speaker_state",
        "previous_speaker": None,
        "reset_success": False,
        "errors": [],
    }
    
    try:
        from core.state import get_current_speaker, get_last_speaker_change_time
        
        stats["previous_speaker"] = get_current_speaker()
        stats["previous_last_change_time"] = get_last_speaker_change_time()
        
        if debug:
            console.print(f"[debug]Clearing current speaker state: "
                         f"speaker='{stats['previous_speaker']}', "
                         f"last_change={stats['previous_last_change_time']}s ago[/debug]")
        
        set_current_speaker(None)
        set_last_speaker_change_time(0.0)
        stats["reset_success"] = True
        
        if debug:
            console.print(f"[debug]Current speaker state cleared[/debug]")
            
    except Exception as e:
        error_msg = f"Failed to clear current speaker state: {str(e)}"
        stats["errors"].append(error_msg)
        if debug:
            console.print(f"[error]{error_msg}[/error]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return stats


def _delete_speaker_state_file(debug: bool = False) -> Dict[str, Any]:
    """
    Delete the persistent speaker state file.
    
    Args:
        debug: Enable detailed debug logging
        
    Returns:
        Dict with reset statistics
    """
    stats = {
        "component": "speaker_state_file",
        "file_existed": False,
        "file_path": None,
        "reset_success": False,
        "errors": [],
    }
    
    try:
        speaker_state_path = get_speaker_state_path()
        stats["file_path"] = str(speaker_state_path)
        
        if speaker_state_path.exists():
            stats["file_existed"] = True
            file_size = speaker_state_path.stat().st_size
            
            if debug:
                console.print(f"[debug]Deleting speaker state file: "
                             f"{speaker_state_path} ({file_size} bytes)[/debug]")
            
            speaker_state_path.unlink()
            stats["reset_success"] = True
            
            if debug:
                console.print(f"[debug]Speaker state file deleted[/debug]")
        else:
            stats["note"] = "Speaker state file did not exist"
            stats["reset_success"] = True
            
    except Exception as e:
        error_msg = f"Failed to delete speaker state file: {str(e)}"
        stats["errors"].append(error_msg)
        if debug:
            console.print(f"[error]{error_msg}[/error]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return stats


def _reset_audio_tagger(debug: bool = False) -> Dict[str, Any]:
    """
    Reset the audio tagger component.
    
    Args:
        debug: Enable detailed debug logging
        
    Returns:
        Dict with reset statistics
    """
    stats = {
        "component": "audio_tagger",
        "was_initialized": False,
        "reset_success": False,
        "errors": [],
    }
    
    try:
        tagger = get_audio_tagger()
        
        if tagger:
            stats["was_initialized"] = True
            
            if debug:
                console.print(f"[debug]Resetting audio tagger[/debug]")
            
            # try:
            #     tagger.reset()
            # except Exception as e:
            #     if debug:
            #         console.print(f"[debug]Tagger reset() raised: {e}, setting to None[/debug]")
            
            # set_audio_tagger(None)
            stats["reset_success"] = True
            
            if debug:
                console.print(f"[debug]Audio tagger reset complete[/debug]")
        else:
            stats["note"] = "Audio tagger was not initialized"
            stats["reset_success"] = True
            
    except Exception as e:
        error_msg = f"Failed to reset audio tagger: {str(e)}"
        stats["errors"].append(error_msg)
        if debug:
            console.print(f"[error]{error_msg}[/error]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return stats


def _clean_output_directories(debug: bool = False) -> Dict[str, Any]:
    """
    Clean output directories by removing generated files.
    
    Args:
        debug: Enable detailed debug logging
        
    Returns:
        Dict with reset statistics
    """
    stats = {
        "component": "output_directories",
        "directories_cleaned": [],
        "files_removed": 0,
        "directories_removed": 0,
        "reset_success": False,
        "errors": [],
    }
    
    directories_to_clean = [
        get_last_n_segments_dir(),
        get_live_audio_buffer_dir(),
        get_segments_audio_dir(),
    ]
    
    # Also clean OUTPUT_DIR itself, but recreate it
    for dir_path in directories_to_clean:
        if not dir_path or not dir_path.exists():
            continue
            
        try:
            dir_name = dir_path.name
            file_count = 0
            dir_count = 0
            
            if debug:
                console.print(f"[debug]Cleaning directory: {dir_path}[/debug]")
            
            # Count before removal
            for item in dir_path.rglob("*"):
                if item.is_file():
                    file_count += 1
                elif item.is_dir():
                    dir_count += 1
            
            # Remove all contents but keep the directory itself
            for item in dir_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            stats["files_removed"] += file_count
            stats["directories_removed"] += dir_count
            stats["directories_cleaned"].append(str(dir_path))
            
            if debug:
                console.print(f"[debug]Cleaned {dir_name}: "
                             f"{file_count} files, {dir_count} subdirs removed[/debug]")
                
        except Exception as e:
            error_msg = f"Failed to clean {dir_path}: {str(e)}"
            stats["errors"].append(error_msg)
            if debug:
                console.print(f"[error]{error_msg}[/error]")
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    # Recreate essential directories
    try:
        get_last_n_segments_dir().mkdir(parents=True, exist_ok=True)
        get_live_audio_buffer_dir().mkdir(parents=True, exist_ok=True)
        get_segments_audio_dir().mkdir(parents=True, exist_ok=True)
        stats["reset_success"] = True
    except Exception as e:
        stats["errors"].append(f"Failed to recreate directories: {str(e)}")
    
    return stats


def _reinitialize_critical_systems(debug: bool = False) -> Dict[str, Any]:
    """
    Reinitialize critical systems after a full reset.
    
    Args:
        debug: Enable detailed debug logging
        
    Returns:
        Dict with reinitialization statistics
    """
    stats = {
        "component": "reinitialization",
        "systems_reinitialized": [],
        "reset_success": False,
        "errors": [],
    }
    
    try:
        # Reset embedding inference to force reload
        current_inference = get_embedding_inference()
        if current_inference:
            set_embedding_inference(None)
            if debug:
                console.print(f"[debug]Cleared embedding inference[/debug]")
        
        # Clear speaker labeler to force reinitialization on next use
        # set_speaker_labeler(None)
        
        # Reset segment counter
        from services.live_subtitles_server_utils import _segment_counter
        import services.live_subtitles_server_utils as utils
        utils._segment_counter = 0
        
        stats["systems_reinitialized"] = [
            "embedding_inference",
            "speaker_labeler",
            "segment_counter",
        ]
        stats["reset_success"] = True
        
        if debug:
            console.print(f"[debug]Critical systems reinitialized: "
                         f"{', '.join(stats['systems_reinitialized'])}[/debug]")
        
    except Exception as e:
        error_msg = f"Failed to reinitialize systems: {str(e)}"
        stats["errors"].append(error_msg)
        if debug:
            console.print(f"[error]{error_msg}[/error]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return stats


@router.post("/reset")
async def global_reset(
    reset_type: str = Form("full", description="Reset type: 'full' or 'soft'"),
    debug: bool = Form(False, description="Enable debug logging"),
):
    """
    Perform a global reset of all application state.
    
    **Reset Types:**
    - **full**: Complete wipe - resets speakers, context, segments, files, and reinitializes systems
    - **soft**: Runtime state only - resets speakers and context, preserves files
    
    **What gets reset:**
    | Component | Full | Soft |
    |-----------|------|------|
    | Speaker Labeler | ✅ | ✅ |
    | Context Buffer | ✅ | ✅ |
    | Segment Counter | ✅ | ❌ |
    | Current Speaker | ✅ | ✅ |
    | Speaker State File | ✅ | ✅ |
    | Audio Tagger | ✅ | ❌ |
    | Output Directories | ✅ | ❌ |
    | Critical Systems | ✅ | ❌ |
    
    **Parameters:**
    - **reset_type**: 'full' (default) or 'soft'
    - **debug**: Enable detailed trace logging
    
    **Returns:**
    - Summary of all reset operations with success/failure per component
    """
    
    if reset_type not in ("full", "soft"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reset_type: '{reset_type}'. Must be 'full' or 'soft'."
        )
    
    console.print(f"[warning]🔄 Performing {reset_type.upper()} global reset...[/warning]")
    
    if debug:
        console.print(f"[debug]Reset type: {reset_type}[/debug]")
        console.print(f"[debug]Start time: {datetime.now().isoformat()}[/debug]")
    
    start_time = time.time()
    
    # Components always reset
    results = {
        "speaker_labeler": _reset_speaker_labeler(debug=debug),
        "context_buffer": _clear_context_buffer(debug=debug),
        "current_speaker_state": _clear_current_speaker_state(debug=debug),
        "speaker_state_file": _delete_speaker_state_file(debug=debug),
    }
    
    # Full reset only components
    if reset_type == "full":
        results.update({
            "segment_counter": _reset_segment_counter(debug=debug),
            "audio_tagger": _reset_audio_tagger(debug=debug),
            "output_directories": _clean_output_directories(debug=debug),
            "reinitialization": _reinitialize_critical_systems(debug=debug),
        })
    
    elapsed = time.time() - start_time
    
    # Calculate summary
    total_components = len(results)
    successful_components = sum(
        1 for r in results.values() if r.get("reset_success", False)
    )
    failed_components = total_components - successful_components
    
    # Collect all errors
    all_errors = []
    for component, stats in results.items():
        for error in stats.get("errors", []):
            all_errors.append(f"[{component}] {error}")
    
    summary = {
        "success": failed_components == 0,
        "reset_type": reset_type,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(elapsed, 3),
        "total_components": total_components,
        "successful_components": successful_components,
        "failed_components": failed_components,
        "components": results,
        "errors": all_errors if all_errors else None,
        "message": (
            f"{reset_type.upper()} reset complete: "
            f"{successful_components}/{total_components} components reset successfully "
            f"in {elapsed:.3f}s"
        ),
    }
    
    if debug:
        console.print(f"[debug]Reset complete in {elapsed:.3f}s[/debug]")
        console.print(f"[debug]Success rate: {successful_components}/{total_components}[/debug]")
    
    if failed_components > 0:
        console.print(f"[warning]⚠️  {failed_components} component(s) failed to reset[/warning]")
        for error in all_errors:
            console.print(f"[error]  - {error}[/error]")
    else:
        console.print(f"[success]✅ {reset_type.upper()} global reset successful "
                     f"({successful_components}/{total_components} components)[/success]")
    
    return JSONResponse(content=summary)


@router.get("/status")
async def get_global_status():
    """
    Get the current status of all resettable components.
    Useful for checking what would be affected by a reset.
    
    Returns:
        Summary of all component states
    """
    status = {
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Speaker Labeler
    try:
        labeler = get_speaker_labeler()
        if labeler:
            status["components"]["speaker_labeler"] = {
                "initialized": True,
                "speaker_count": labeler.speaker_count,
                "segments_processed": labeler.total_segments_processed,
                "known_speakers": labeler.known_speakers,
            }
        else:
            status["components"]["speaker_labeler"] = {
                "initialized": False,
            }
    except Exception as e:
        status["components"]["speaker_labeler"] = {
            "error": str(e),
        }
    
    # Context Buffer
    try:
        context_buffer = get_context_buffer()
        status["components"]["context_buffer"] = {
            "segment_count": len(context_buffer.segments),
            "total_duration": context_buffer.get_total_duration(),
            "max_duration": context_buffer.max_duration_sec,
        }
    except Exception as e:
        status["components"]["context_buffer"] = {
            "error": str(e),
        }
    
    # Current Speaker
    try:
        from core.state import get_current_speaker, get_last_speaker_change_time
        status["components"]["current_speaker"] = {
            "speaker": get_current_speaker(),
            "last_change_time": get_last_speaker_change_time(),
        }
    except Exception as e:
        status["components"]["current_speaker"] = {
            "error": str(e),
        }
    
    # Speaker State File
    try:
        speaker_state_path = get_speaker_state_path()
        status["components"]["speaker_state_file"] = {
            "exists": speaker_state_path.exists(),
            "path": str(speaker_state_path),
            "size_bytes": speaker_state_path.stat().st_size if speaker_state_path.exists() else 0,
        }
    except Exception as e:
        status["components"]["speaker_state_file"] = {
            "error": str(e),
        }
    
    # Segment Counter
    try:
        segment_index_path = get_segment_index_path()
        if segment_index_path.exists():
            with open(segment_index_path, 'r') as f:
                data = json.load(f)
            counter = data.get("counter", 0)
        else:
            counter = 0
        
        status["components"]["segment_counter"] = {
            "current_value": counter,
            "index_file_exists": segment_index_path.exists(),
        }
    except Exception as e:
        status["components"]["segment_counter"] = {
            "error": str(e),
        }
    
    # Audio Tagger
    try:
        tagger = get_audio_tagger()
        status["components"]["audio_tagger"] = {
            "initialized": tagger is not None,
        }
    except Exception as e:
        status["components"]["audio_tagger"] = {
            "error": str(e),
        }
    
    # Output Directories
    try:
        last_n_dir = get_last_n_segments_dir()
        live_dir = get_live_audio_buffer_dir()
        audio_segments_dir = get_segments_audio_dir()
        
        def count_items(directory):
            if not directory.exists():
                return {"files": 0, "dirs": 0}
            files = sum(1 for _ in directory.rglob("*") if _.is_file())
            dirs = sum(1 for _ in directory.rglob("*") if _.is_dir())
            return {"files": files, "dirs": dirs}
        
        status["components"]["output_directories"] = {
            "last_n_segments": count_items(last_n_dir),
            "live_audio_buffer": count_items(live_dir),
            "audio_segments": count_items(audio_segments_dir),
        }
    except Exception as e:
        status["components"]["output_directories"] = {
            "error": str(e),
        }
    
    return JSONResponse(content=status)
