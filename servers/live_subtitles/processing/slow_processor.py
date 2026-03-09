import logging
import json
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from segment_speaker_labeler import SegmentSpeakerLabeler
# from segment_emotion_classifier import SegmentEmotionClassifier

logger = logging.getLogger(__name__)

# Global loaded models
logger.info("Loading speaker diarization model...")
_speaker_labeler = SegmentSpeakerLabeler()

# logger.info("Loading emotion classification model...")
# _emotion_classifier = SegmentEmotionClassifier(device=-1)  # cpu; change in prod


def get_speaker_labeler() -> SegmentSpeakerLabeler:
    global _speaker_labeler
    return _speaker_labeler


# def get_emotion_classifier() -> SegmentEmotionClassifier:
#     global _emotion_classifier
#     return _emotion_classifier


def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """Convert raw little-endian int16 PCM → float32 waveform in [-1, 1]"""
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
    Blocking slow-path function – run in executor / background thread.
    Returns (speaker_result_dict, emotion_result_dict)
    """
    utt_short = f"utt{utterance_id} seg{segment_idx}/{segment_num}"

    logger.info(
        "[slow start] %s | curr=%d bytes | prev=%s bytes | sr=%d Hz",
        utt_short,
        len(curr_pcm),
        len(prev_pcm) if prev_pcm else "None",
        sample_rate
    )

    speaker_result = {
        "is_same_speaker_as_prev": None,
        "similarity_prev": None,
        "cluster_speakers": None,
    }
    emotion_result = {
        "emotion_top_label": None,
        "emotion_top_score": None,
        "emotion_all": [],
    }

    success = True

    # ── Speaker analysis ───────────────────────────────────────────────
    speaker_labeler = get_speaker_labeler()

    curr_wave = pcm_bytes_to_float32(curr_pcm)
    logger.debug("%s | converted curr waveform: %.1fs", utt_short, len(curr_wave)/sample_rate)

    prev_wave = None
    if prev_pcm is not None and len(prev_pcm) > 0:
        prev_wave = pcm_bytes_to_float32(prev_pcm)
        logger.debug("%s | converted prev waveform: %.1fs", utt_short, len(prev_wave)/sample_rate)

    is_same = False
    sim = None
    clusters = None

    if prev_wave is not None:
        try:
            clusters = speaker_labeler.cluster_segments([prev_wave, curr_wave])
            if len(clusters) == 2:
                clusters = [str(c) for c in clusters]  # ensure serializable
                sim = speaker_labeler.similarity(curr_wave, prev_wave)
                is_same = speaker_labeler.is_same_speaker(curr_wave, prev_wave)
                logger.info(
                    "%s | speaker → same=%s | sim=%.3f | clusters=%s",
                    utt_short, is_same, sim or -1, clusters
                )
            else:
                logger.warning("%s | unexpected cluster count: %d", utt_short, len(clusters))
        except Exception as e:
            logger.warning("%s | speaker analysis failed → using fallback", utt_short, exc_info=True)

    speaker_result.update({
        "is_same_speaker_as_prev": is_same,
        "similarity_prev": round(float(sim), 3) if sim is not None else None,
        "cluster_speakers": clusters,
    })

    # # ── Emotion classification ─────────────────────────────────────────
    # emotion_clf = get_emotion_classifier()
    #
    # emo_results = emotion_clf.classify(curr_pcm, sample_rate)
    # logger.debug("%s | raw emotion results: %d entries", utt_short, len(emo_results or []))
    #
    # top_label = None
    # top_score = None
    #
    # if emo_results:
    #     top = max(emo_results, key=lambda x: x.get("score", -float("inf")))
    #     top_label = top.get("label")
    #     top_score = top.get("score")
    #     logger.info(
    #         "%s | emotion → top=%s (%.3f) | %d classes",
    #         utt_short, top_label, top_score or 0.0, len(emo_results)
    #     )
    # else:
    #     logger.warning("%s | emotion classifier returned empty result", utt_short)
    #
    # emotion_result.update({
    #     "emotion_top_label": top_label,
    #     "emotion_top_score": round(float(top_score), 3) if top_score is not None else None,
    #     "emotion_all": emo_results or [],
    # })

    # Final outcome log
    if success:
        logger.info(
            "[slow done]  %s | speaker same=%s | sim=%.3f | emotion=%s (%.3f)",
            utt_short,
            speaker_result["is_same_speaker_as_prev"],
            speaker_result["similarity_prev"] or -1,
            emotion_result["emotion_top_label"] or "N/A",
            emotion_result["emotion_top_score"] or 0.0
        )
    else:
        logger.warning(
            "[slow FAIL]  %s | speaker=%s | emotion=%s",
            utt_short,
            speaker_result["is_same_speaker_as_prev"],
            emotion_result["emotion_top_label"] or "N/A"
        )

    return speaker_result, emotion_result
