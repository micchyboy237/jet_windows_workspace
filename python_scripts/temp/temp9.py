import torch
from pathlib import Path
from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

DEFAULT_AUDIO = str(
    Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
)

audio_file = DEFAULT_AUDIO

# ==================================================================
# FEATURE 4: Newer model — pyannote/segmentation-3.0.0
# ==================================================================
print("=== Feature 4: Newer Segmentation Model ===\n")

# The 3.0.0 model uses powerset mode — its output is already binarized,
# so onset and offset are fixed at 0.5 and cannot be tuned.
# Only min_duration_on and min_duration_off remain as free parameters.

pipeline_v3 = VoiceActivityDetection(
    segmentation="pyannote/segmentation-3.0.0",
    # token="hf_your_token_here",
)

defaults_v3 = pipeline_v3.default_parameters()
print(f"v3.0 default params: {defaults_v3}")
# → {'min_duration_on': 0.0, 'min_duration_off': 0.0}
# onset and offset are NOT tunable in powerset mode

pipeline_v3.instantiate(defaults_v3)
speech_v3 = pipeline_v3(audio_file)
print(f"v3.0 regions found : {len(list(speech_v3.itertracks()))}\n")
