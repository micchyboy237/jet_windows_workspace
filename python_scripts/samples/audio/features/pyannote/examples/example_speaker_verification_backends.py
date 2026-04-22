# example_speaker_verification_backends.py
"""
All Four Embedding Backends Side by Side
=========================================
Goal: Show how to load and use each of the four backend classes,
      and understand when to choose each one.

Quick comparison:
┌──────────────────────────────────────────┬──────────────────────────────────────┐
│ Backend                                  │ When to use                          │
├──────────────────────────────────────────┼──────────────────────────────────────┤
│ PyannoteAudioPretrainedSpeakerEmbedding  │ Default — tight pyannote integration │
│ SpeechBrainPretrainedSpeakerEmbedding    │ ECAPA-TDNN, strong SOTA model        │
│ NeMoPretrainedSpeakerEmbedding           │ NVIDIA TitaNet, large-scale models   │
│ ONNXWeSpeakerPretrainedSpeakerEmbedding  │ Portable ONNX, no heavy framework    │
└──────────────────────────────────────────┴──────────────────────────────────────┘

All four share the same __call__ signature:
  embeddings = backend(waveforms)            → shape (batch, dim)
  embeddings = backend(waveforms, masks=m)   → shape (batch, dim), masked
"""

import numpy as np
import torch

SR = 16000
DURATION = 3   # seconds
BATCH    = 2

# Synthetic waveforms: (batch_size=2, channels=1, samples=48000)
fake_waveforms = torch.randn(BATCH, 1, SR * DURATION)

# Soft masks: (batch_size=2, num_frames)
# Values between 0 and 1 — 1 means "this frame is definitely speech"
NUM_FRAMES = 300
fake_masks = torch.ones(BATCH, NUM_FRAMES)
fake_masks[0, 200:] = 0.0   # last third of clip 1 is silence


def probe_backend(name, backend):
    """Print key properties and run a quick embedding extraction."""
    print(f"\n{'─' * 55}")
    print(f"  Backend : {name}")
    print(f"{'─' * 55}")
    print(f"  sample_rate      : {backend.sample_rate}")
    print(f"  dimension        : {backend.dimension}")
    print(f"  metric           : {backend.metric}")
    print(f"  min_num_samples  : {backend.min_num_samples}  "
          f"({backend.min_num_samples / backend.sample_rate * 1000:.1f} ms)")

    embs = backend(fake_waveforms)
    print(f"  Output shape     : {embs.shape}")
    print(f"  Any NaN?         : {np.any(np.isnan(embs))}")
    print(f"  Norm (clip 0)    : {np.linalg.norm(embs[0]):.4f}")

    # With masks
    embs_masked = backend(fake_waveforms, masks=fake_masks)
    print(f"  Masked shape     : {embs_masked.shape}")
    nan_masked = np.any(np.isnan(embs_masked))
    print(f"  Any NaN (masked) : {nan_masked}")


# ==================================================================
# BACKEND 1: PyannoteAudio
# ==================================================================
print("=" * 55)
print("BACKEND 1: PyannoteAudioPretrainedSpeakerEmbedding")
print("=" * 55)
print("""
Model source  : HuggingFace ("pyannote/embedding" or custom fine-tune)
Framework     : PyTorch (pyannote.audio Model class)
Masking       : Passes weights= to the model's forward() call
Best for      : Tight pyannote ecosystem integration, fine-tuning
Install       : pip install pyannote.audio
""")
try:
    from pyannote.audio.pipelines.speaker_verification import (
        PyannoteAudioPretrainedSpeakerEmbedding,
    )
    backend_pya = PyannoteAudioPretrainedSpeakerEmbedding(
        "pyannote/embedding",
        # token="hf_your_token_here",
    )
    probe_backend("PyannoteAudio", backend_pya)
except Exception as e:
    print(f"  (Skipped: {type(e).__name__}: {e})")


# ==================================================================
# BACKEND 2: SpeechBrain
# ==================================================================
print("\n" + "=" * 55)
print("BACKEND 2: SpeechBrainPretrainedSpeakerEmbedding")
print("=" * 55)
print("""
Model source  : HuggingFace ("speechbrain/spkrec-ecapa-voxceleb", etc.)
Framework     : SpeechBrain (EncoderClassifier)
Masking       : Passes wav_lens= (relative lengths 0–1) to encode_batch()
Best for      : ECAPA-TDNN — strong SOTA speaker model
Install       : pip install speechbrain

Tip: Use "@revision" suffix to pin a specific model version:
     "speechbrain/spkrec-ecapa-voxceleb@v3.0.0"
""")
try:
    from pyannote.audio.pipelines.speaker_verification import (
        SpeechBrainPretrainedSpeakerEmbedding,
    )
    backend_sb = SpeechBrainPretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        # token="hf_your_token_here",
        # cache_dir="/tmp/speechbrain",
    )
    probe_backend("SpeechBrain", backend_sb)
except Exception as e:
    print(f"  (Skipped: SpeechBrain not installed or model unavailable: {e})")


# ==================================================================
# BACKEND 3: NeMo
# ==================================================================
print("\n" + "=" * 55)
print("BACKEND 3: NeMoPretrainedSpeakerEmbedding")
print("=" * 55)
print("""
Model source  : NVIDIA NGC / HuggingFace ("nvidia/speakerverification_en_titanet_large")
Framework     : NVIDIA NeMo (EncDecSpeakerLabelModel)
Masking       : Interpolates mask to waveform length, pads masked sequences
Best for      : Large-scale NVIDIA models, TitaNet architecture
Install       : pip install nemo_toolkit[asr]

Note: NeMo's .to(device) reloads from scratch — use device= at init time
      instead of calling .to() afterwards when possible.
""")
try:
    from pyannote.audio.pipelines.speaker_verification import (
        NeMoPretrainedSpeakerEmbedding,
    )
    backend_nemo = NeMoPretrainedSpeakerEmbedding(
        "nvidia/speakerverification_en_titanet_large",
        device=torch.device("cpu"),
    )
    probe_backend("NeMo", backend_nemo)
except Exception as e:
    print(f"  (Skipped: NeMo not installed or model unavailable: {e})")


# ==================================================================
# BACKEND 4: WeSpeaker (ONNX)
# ==================================================================
print("\n" + "=" * 55)
print("BACKEND 4: ONNXWeSpeakerPretrainedSpeakerEmbedding")
print("=" * 55)
print("""
Model source  : HuggingFace ("hbredin/wespeaker-voxceleb-resnet34-LM")
                or local path to a .onnx file
Framework     : ONNX Runtime (no PyTorch needed at inference time)
Input         : fbank features (not raw waveforms) — computed internally
Masking       : Masks are applied to the fbank feature frames
Best for      : Portable deployment, no heavy framework dependencies
Install       : pip install onnxruntime

Quirk: compute_fbank() is called on raw waveforms first,
       then the ONNX session receives (batch, frames, 80) fbank features.
""")
try:
    from pyannote.audio.pipelines.speaker_verification import (
        ONNXWeSpeakerPretrainedSpeakerEmbedding,
    )
    backend_onnx = ONNXWeSpeakerPretrainedSpeakerEmbedding(
        "hbredin/wespeaker-voxceleb-resnet34-LM",
        # token="hf_your_token_here",
        # cache_dir="/tmp/wespeaker",
    )
    probe_backend("WeSpeaker ONNX", backend_onnx)

    # Show fbank feature extraction explicitly
    print(f"\n  Fbank feature demo:")
    fbank = backend_onnx.compute_fbank(fake_waveforms[:1])
    print(f"    Input waveform : {fake_waveforms[:1].shape}")
    print(f"    fbank output   : {fbank.shape}  (batch, frames, 80 mel bins)")
    print(f"    min_num_frames : {backend_onnx.min_num_frames}")
except Exception as e:
    print(f"  (Skipped: onnxruntime not installed or model unavailable: {e})")


# ==================================================================
# ROUTER: PretrainedSpeakerEmbedding()
# ==================================================================
print("\n" + "=" * 55)
print("ROUTER: PretrainedSpeakerEmbedding()")
print("=" * 55)
print("""
This is the function you should use in most cases.
It inspects the model name string and returns the right backend.
""")

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

routing_table = [
    ("pyannote/embedding",                          "PyannoteAudio"),
    ("speechbrain/spkrec-ecapa-voxceleb",           "SpeechBrain"),
    ("nvidia/speakerverification_en_titanet_large", "NeMo"),
    ("hbredin/wespeaker-voxceleb-resnet34-LM",      "WeSpeaker ONNX"),
]

print(f"  {'Model string':50s}  →  Expected backend")
for model_str, expected in routing_table:
    # Show routing logic without actually downloading
    if "pyannote" in model_str:
        resolved = "PyannoteAudioPretrainedSpeakerEmbedding"
    elif "speechbrain" in model_str:
        resolved = "SpeechBrainPretrainedSpeakerEmbedding"
    elif "nvidia" in model_str:
        resolved = "NeMoPretrainedSpeakerEmbedding"
    elif "wespeaker" in model_str:
        resolved = "ONNXWeSpeakerPretrainedSpeakerEmbedding"
    else:
        resolved = "PyannoteAudioPretrainedSpeakerEmbedding (fallback)"
    print(f"  {model_str:50s}  →  {resolved}")
