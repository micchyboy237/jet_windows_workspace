"""
Slow / background processing: speaker diarization + emotion classification
(placeholder / dummy implementations for now)
"""
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

_speaker_labeler = None
_emotion_classifier = None


def get_speaker_labeler():
    global _speaker_labeler
    if _speaker_labeler is None:
        logger.info("Lazy loading speaker diarization model...")
        _speaker_labeler = "dummy-labeler"
    return _speaker_labeler


def get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        logger.info("Lazy loading emotion classification model...")
        _emotion_classifier = "dummy-emotion"
    return _emotion_classifier


def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """Convert raw int16 PCM → float32 normalized waveform [-1, 1]"""
    arr = np.frombuffer(pcm, dtype=np.int16)
    if arr.size == 0:
        raise ValueError("Empty PCM buffer")
    return arr.astype(np.float32) / 32768.0


def process_slow(
    curr_pcm: bytes,
    prev_pcm: Optional[bytes],
    sample_rate: int,
    utterance_id: str,
    segment_idx: int,
    segment_num: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Blocking slow-path function – run in executor_slow.
    Returns (speaker_result_dict, emotion_result_dict)
    """
    try:
        # Placeholder speaker logic
        speaker_labeler = get_speaker_labeler()
        curr_wave = pcm_bytes_to_float32(curr_pcm)
        prev_wave = pcm_bytes_to_float32(prev_pcm) if prev_pcm and len(prev_pcm) > 0 else None

        is_same_speaker = False
        similarity = None
        clusters = None

        if prev_wave is not None:
            similarity = 0.84  # dummy
            is_same_speaker = True
            clusters = ["SPEAKER_00", "SPEAKER_00"]

        speaker_result = {
            "is_same_speaker_as_prev": is_same_speaker,
            "similarity_prev": round(similarity, 3) if similarity is not None else None,
            "cluster_speakers": clusters,
        }

        # Placeholder emotion logic
        emotion_clf = get_emotion_classifier()
        emo_results = [
            {"label": "neutral", "score": 0.91},
            {"label": "happy", "score": 0.05},
            {"label": "angry", "score": 0.02},
        ]
        top_emotion = max(emo_results, key=lambda x: x["score"], default=None)
        top_label = top_emotion["label"] if top_emotion else None
        top_score = top_emotion["score"] if top_emotion else None

        emotion_result = {
            "emotion_top_label": top_label,
            "emotion_top_score": round(top_score, 3) if top_score is not None else None,
            "emotion_all": emo_results,
        }

        logger.info(
            "[slow] utt %d | speaker same=%s | emotion=%s (%.2f)",
            utterance_id,
            is_same_speaker,
            top_label or "N/A",
            top_score or 0.0
        )

        return speaker_result, emotion_result

    except Exception as e:
        logger.exception(f"[slow] processing failed for utterance {utterance_id}")
        return (
            {"is_same_speaker_as_prev": None, "similarity_prev": None, "cluster_speakers": None},
            {"emotion_top_label": None, "emotion_top_score": None, "emotion_all": []}
        )