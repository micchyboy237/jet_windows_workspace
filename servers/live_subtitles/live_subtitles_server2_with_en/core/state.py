"""
Global application state management.
Centralizes all module-level singletons and state variables
to avoid circular imports and provide clean access patterns.
"""
import json
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch
from pyannote.audio import Inference, Model
from config import OUTPUT_DIR
from services.segment_speaker_labeler import SegmentSpeakerLabeler
from services.audio_language_detector import AudioLanguageDetector
from services.live_subtitles_server_utils import load_segment_counter

N_SEGMENT_RESULTS = 50
LAST_N_SEGMENTS_DIR = OUTPUT_DIR / f"last_{N_SEGMENT_RESULTS}_segments"
LAST_N_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_AUDIO_BUFFER_DIR = OUTPUT_DIR
LIVE_AUDIO_BUFFER_DIR.mkdir(parents=True, exist_ok=True)
SPEAKER_STATE_PATH = OUTPUT_DIR / "speaker_state.json"
_SEGMENT_INDEX_PATH = LAST_N_SEGMENTS_DIR / "_segment_index.json"

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcribe_worker")
active_connections: dict[str, object] = {}

from services.audio_context_buffer import AudioContextBuffer
context_buffer = AudioContextBuffer(max_duration_sec=30.0, sample_rate=16000)

prev_end_sec: Optional[float] = None
prev_vad_reason = None

_speaker_labeler: Optional[SegmentSpeakerLabeler] = None
_embedding_model: Optional[Model] = None
_embedding_inference: Optional[Inference] = None
_current_speaker: Optional[str] = None
_last_speaker_change_time: float = 0.0

audio_language_detector = None
_audio_tagger = None  # AudioTagger instance
_segment_counter: Optional[int] = None

# ============================================================================
# Segment Counter
# ============================================================================

def get_segment_counter() -> int:
    """Get or initialize the segment counter."""
    global _segment_counter
    if _segment_counter is None:
        _segment_counter = load_segment_counter(_SEGMENT_INDEX_PATH)
    return _segment_counter

def get_segment_index_path() -> Path:
    """Get the segment index file path."""
    return _SEGMENT_INDEX_PATH

def get_n_segment_results() -> int:
    """Get the number of segment results to keep."""
    return N_SEGMENT_RESULTS

def get_last_n_segments_dir() -> Path:
    """Get the last N segments directory."""
    return LAST_N_SEGMENTS_DIR

def get_live_audio_buffer_dir() -> Path:
    """Get the live audio buffer directory."""
    return LIVE_AUDIO_BUFFER_DIR

def get_speaker_state_path() -> Path:
    """Get the speaker state file path."""
    return SPEAKER_STATE_PATH

# ============================================================================
# Executor and Connections
# ============================================================================

def get_executor() -> ThreadPoolExecutor:
    """Get the thread pool executor."""
    return executor

def get_context_buffer() -> AudioContextBuffer:
    """Get the audio context buffer."""
    return context_buffer

def get_active_connections() -> dict:
    """Get active WebSocket connections."""
    return active_connections

# ============================================================================
# VAD State
# ============================================================================

def get_prev_state() -> tuple:
    """Get previous VAD state."""
    return prev_end_sec, prev_vad_reason

def set_prev_state(end_sec: Optional[float], vad_reason) -> None:
    """Set previous VAD state."""
    global prev_end_sec, prev_vad_reason
    prev_end_sec = end_sec
    prev_vad_reason = vad_reason

# ============================================================================
# Speaker State
# ============================================================================

def get_current_speaker() -> Optional[str]:
    """Get current speaker label."""
    return _current_speaker

def set_current_speaker(speaker: Optional[str]) -> None:
    """Set current speaker label."""
    global _current_speaker
    _current_speaker = speaker

def get_last_speaker_change_time() -> float:
    """Get last speaker change timestamp."""
    return _last_speaker_change_time

def set_last_speaker_change_time(timestamp: float) -> None:
    """Set last speaker change timestamp."""
    global _last_speaker_change_time
    _last_speaker_change_time = timestamp

def get_speaker_labeler() -> Optional[SegmentSpeakerLabeler]:
    """Get the current speaker labeler instance."""
    return _speaker_labeler

def set_speaker_labeler(labeler: SegmentSpeakerLabeler) -> None:
    """Set the speaker labeler instance."""
    global _speaker_labeler
    _speaker_labeler = labeler

def get_embedding_inference() -> Optional[Inference]:
    """Get the embedding inference instance."""
    return _embedding_inference

def set_embedding_inference(inference: Inference) -> None:
    """Set the embedding inference instance."""
    global _embedding_inference
    _embedding_inference = inference

def get_speaker_diarization() -> Dict:
    """Get current speaker diarization summary with speaker list support."""
    labeler = get_speaker_labeler()
    if labeler is None:
        return {
            "current_speaker": None,
            "known_speakers": [],
            "speaker_count": 0,
            "speakers_info": {},
            "total_segments_processed": 0,
        }

    all_info = labeler.get_all_speakers_info()
    sorted_speakers = sorted(
        all_info.items(),
        key=lambda x: x[1].get("last_seen", 0),
        reverse=True,
    )
    return {
        "current_speaker": get_current_speaker(),
        "total_segments_processed": labeler.total_segments_processed,
        "known_speakers": labeler.known_speakers,
        "speaker_count": labeler.speaker_count,
        "speakers_info": dict(sorted_speakers),
    }

def save_speaker_state() -> None:
    """Persist the current speaker labeler state to disk."""
    labeler = _speaker_labeler
    if labeler is None:
        return
    try:
        state = labeler.to_dict()
        with open(SPEAKER_STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

# ============================================================================
# Audio Language Detector
# ============================================================================

def get_audio_language_detector():
    """Get the audio language detector."""
    return audio_language_detector

def set_audio_language_detector(detector) -> None:
    """Set the audio language detector."""
    global audio_language_detector
    audio_language_detector = detector

# ============================================================================
# Audio Tagger (NEW)
# ============================================================================

def get_audio_tagger():
    """
    Get the audio tagger instance.
    
    Returns:
        AudioTagger instance or None if not initialized
    """
    return _audio_tagger

def set_audio_tagger(tagger) -> None:
    """
    Set the audio tagger instance.
    
    Args:
        tagger: AudioTagger instance
    """
    global _audio_tagger
    _audio_tagger = tagger
