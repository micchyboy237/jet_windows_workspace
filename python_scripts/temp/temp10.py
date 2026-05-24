# example_vad_advanced.py
"""
Advanced VAD — Hooks, Parameter Tuning, and F-score Mode
=========================================================
Goal: Show the less-obvious controls available in VoiceActivityDetection.

Covers:
  1. Progress hook — watch segmentation unfold in real time
  2. Manual parameter tuning — onset, offset, duration filters
  3. fscore=True — switch the optimisation target from DER to F-measure
  4. Using a newer model (pyannote/segmentation-3.0.0)
  5. Batch size control for faster GPU processing
  6. Passing a raw waveform tensor instead of a file path
"""

import numpy as np
from pathlib import Path
import torch
from pyannote.audio.pipelines.voice_activity_detection import VoiceActivityDetection

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

DEFAULT_AUDIO = str(
    Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
)

audio_file = DEFAULT_AUDIO

# ==================================================================
# FEATURE 1: Progress hook
# ==================================================================
print("=== Feature 1: Progress Hook ===\n")

# The hook is called after each major pipeline step with:
#   step_name     → name of the stage (str)
#   step_artifact → what was produced (SlidingWindowFeature, Annotation, ...)
#   completed     → batches done so far (int, only during long steps)
#   total         → total batches in this step (int, only during long steps)

def my_hook(step_name, step_artifact, file=None, completed=None, total=None):
    if completed is not None and total is not None:
        pct = 100 * completed / total
        print(f"  [{step_name}] {completed}/{total} ({pct:.0f}%)")
    else:
        info = ""
        if hasattr(step_artifact, "data"):
            info = f"shape={step_artifact.data.shape}"
        elif hasattr(step_artifact, "shape"):
            info = f"shape={step_artifact.shape}"
        print(f"  ✓ {step_name}  {info}")

pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
pipeline.instantiate(pipeline.default_parameters())

speech = pipeline(audio_file, hook=my_hook)
print(f"\nFound {len(list(speech.itertracks()))} speech regions with hook.\n")


# ==================================================================
# FEATURE 2: Manual parameter tuning
# ==================================================================
print("=== Feature 2: Parameter Tuning ===\n")

# Parameters and what they control:
#
#   onset  (float 0–1):
#     The model's speech probability must RISE ABOVE this to start a region.
#     Higher onset → stricter, fewer false alarms, may miss soft speech.
#
#   offset (float 0–1):
#     The model's speech probability must FALL BELOW this to end a region.
#     Lower offset → speech regions extend further into quiet sections.
#     offset < onset is normal — creates hysteresis (avoids rapid switching).
#
#   min_duration_on (float, seconds):
#     Delete any detected speech region shorter than this.
#     Useful for removing noise spikes labelled as speech.
#
#   min_duration_off (float, seconds):
#     Fill any silence gap shorter than this.
#     Useful for merging closely-spaced words into one region.

custom_params = {
    "onset": 0.8,            # stricter — only high-confidence speech
    "offset": 0.4,           # generous end — don't cut off word endings
    "min_duration_on": 0.2,  # ignore blips shorter than 0.2s
    "min_duration_off": 0.1, # fill pauses shorter than 0.1s
}

pipeline.instantiate(custom_params)
speech_custom = pipeline(audio_file)

print("Default params speech regions :", len(list(
    VoiceActivityDetection(segmentation="pyannote/segmentation")
    .apply.__func__  # just counting — pipeline already set above
    if False else speech.itertracks()  # use earlier result
)))
print("Custom params  speech regions :", len(list(speech_custom.itertracks())))
print()


# ==================================================================
# FEATURE 3: fscore=True — optimise for F-measure instead of DER
# ==================================================================
print("=== Feature 3: F-score Mode ===\n")

# By default the pipeline minimises DER (Detection Error Rate).
# Setting fscore=True switches the optimisation target to F-measure
# (precision × recall harmonic mean), which is better when you care
# equally about not missing speech AND not adding false alarms.

pipeline_fscore = VoiceActivityDetection(
    segmentation="pyannote/segmentation",
    fscore=True,
    # token="hf_your_token_here",
)
pipeline_fscore.instantiate(pipeline_fscore.default_parameters())

speech_fscore = pipeline_fscore(audio_file)

print(f"Optimisation target : {'F-measure (maximize)' if pipeline_fscore.fscore else 'DER (minimize)'}")
print(f"get_direction()     : {pipeline_fscore.get_direction()}")
print(f"get_metric() type   : {type(pipeline_fscore.get_metric()).__name__}")
print(f"Regions found       : {len(list(speech_fscore.itertracks()))}\n")


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


# ==================================================================
# FEATURE 5: Batch size — faster on GPU
# ==================================================================
print("=== Feature 5: Batch Size Control ===\n")

# inference_kwargs are forwarded straight to pyannote.audio.Inference.
# The most useful one is batch_size — how many audio windows to process
# in parallel. Larger = faster on GPU, but uses more VRAM.

pipeline_batched = VoiceActivityDetection(
    segmentation="pyannote/segmentation",
    batch_size=32,   # ← passed to Inference via **inference_kwargs
    # token="hf_your_token_here",
)
pipeline_batched.instantiate(pipeline_batched.default_parameters())
speech_batched = pipeline_batched(audio_file)
print(f"Batched (bs=32) regions: {len(list(speech_batched.itertracks()))}\n")


# ==================================================================
# FEATURE 6: Raw waveform tensor as input
# ==================================================================
print("=== Feature 6: Waveform Tensor Input ===\n")

# Instead of a file path you can pass a dict with:
#   "waveform"    → torch.Tensor of shape (channels, samples)
#   "sample_rate" → int, e.g. 16000

sample_rate = 16000
duration_sec = 5
fake_waveform = torch.randn(1, sample_rate * duration_sec)   # 5 s of noise

audio_dict = {
    "uri": "synthetic_audio",       # optional but useful for logging
    "waveform": fake_waveform,
    "sample_rate": sample_rate,
}

pipeline_w = VoiceActivityDetection(segmentation="pyannote/segmentation")
pipeline_w.instantiate(pipeline_w.default_parameters())
speech_waveform = pipeline_w(audio_dict)

print(f"Input  : waveform tensor {fake_waveform.shape}, sr={sample_rate}")
print(f"Output : {len(list(speech_waveform.itertracks()))} speech regions")
for turn, _, label in speech_waveform.itertracks(yield_label=True):
    print(f"  [{turn.start:.3f}s → {turn.end:.3f}s]  {label}")
