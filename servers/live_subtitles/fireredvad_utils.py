# servers\live_subtitles\fireredvad_utils.py
from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig


def load_vad(vad_model_dir: str) -> FireRedStreamVad:
    """Load FireRed Stream-VAD with stronger tuning for continuous Japanese speech."""
    config = FireRedStreamVadConfig(
        use_gpu=False,
        smooth_window_size=9,       # even smoother
        speech_threshold=0.40,      # more sensitive onset
        pad_start_frame=12,
        min_speech_frame=20,        # significantly reduce false micro-starts
        max_speech_frame=1600,      # ~16s
        min_silence_frame=30,       # need clearer silence to end
        chunk_max_frame=30000,
    )
    vad = FireRedStreamVad.from_pretrained(vad_model_dir, config)
    vad.set_mode(1)
    return vad