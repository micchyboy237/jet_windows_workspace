# Whisper CTranslate2 Models – n_mels Reference & Usage Guide

This README lists all official OpenAI Whisper models that use **`n_mels=80`** (the original Whisper architecture) and **`n_mels=128`** (large-v3 and newer), with direct support for **CTranslate2 `int8_float16`** quantization.

`int8_float16` = INT8 weights + FP16 activations → ~45 % less VRAM/RAM, 1.5–2× faster on GPU, negligible accuracy loss.

All models below can be:
- Auto-downloaded & converted by `faster-whisper`
- Manually converted once with `ct2-transformers-converter --quantization int8_float16`
- Used on **Mac M1 (CPU)** or **Windows + GTX 1660 (CUDA)**

## Model Table

| Model Size   | n_mels | Parameters | English-Only | Multilingual HF ID                     | Approx. VRAM (int8_float16) | Recommended Use Case                     |
|--------------|--------|------------|--------------|----------------------------------------|-----------------------------|------------------------------------------|
| `tiny`       | 80     | 39 M       | `tiny.en`    | `openai/whisper-tiny`                  | ~80 MB                      | Fast prototyping, low-resource devices   |
| `base`       | 80     | 74 M       | `base.en`    | `openai/whisper-base`                  | ~120 MB                     | Testing, CI, lightweight servers         |
| `small`      | 80     | 244 M      | `small.en`   | `openai/whisper-small`                 | ~350 MB                     | Good accuracy/speed balance              |
| `medium`     | 80     | 769 M      | `medium.en`  | `openai/whisper-medium`                | ~1.1 GB                     | High-quality transcription               |
| `large`      | 80     | 1.55 B     | —            | `openai/whisper-large`                 | ~2.2 GB                     | Legacy large (pre-v2)                    |
| `large-v2`   | 80     | 1.55 B     | —            | `openai/whisper-large-v2`              | ~2.2 GB                     | Best pre-v3 multilingual model           |
| `large-v3`   | 128    | 1.55 B     | —            | `openai/whisper-large-v3`              | ~2.9 GB                     | Current best accuracy (2024–2025)        |
| `large-v3-turbo` | 128 | 809 M      | —            | `openai/whisper-large-v3-turbo`        | ~1.8 GB                     | Faster distilled v3                      |
| `distil-large-v3` | 128 | ~756 M    | —            | `distil-whisper/distil-large-v3`       | ~1.7 GB                     | Fast distillation, excellent speed/quality |

> **Rule of thumb**  
> - Use any model with `n_mels=80` if you need maximum compatibility or are mixing with older pipelines.  
> - Use `large-v3` / `large-v3-turbo` (`n_mels=128`) for state-of-the-art accuracy.

## Quick Start Examples (int8_float16)

### 1. Install (once)

```bash
pip install faster-whisper rich tqdm
```

### 2. Single-file transcription (GPU – GTX 1660)

```python
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import track

console = Console()

model = WhisperModel(
    "large-v2",                  # n_mels=80 → ~2.2 GB VRAM (int8_float16)
    # "large-v3",                # n_mels=128 → ~2.9 GB VRAM
    device="cuda",
    compute_type="int8_float16"  # ← hybrid quantization
)

segments, info = model.transcribe(
    "audio.mp3",
    beam_size=5,
    language="en",
    vad_filter=True
)

console.print(f"[bold green]Language:[/] {info.language} ({info.language_probability:.2f})")
for segment in track(segments, description="Transcribing..."):
    console.print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### 3. CPU-only (Mac M1 – fastest quantized mode)

```python
model = WhisperModel(
    "medium",                    # n_mels=80, fits easily in RAM
    device="cpu",
    compute_type="int8_float16"  # or "int8" for pure INT8 (even smaller)
)
```

### 4. Manual conversion (once) → reuse forever

```bash
ct2-transformers-converter \
  --model openai/whisper-large-v2 \
  --output_dir whisper-large-v2-ct2-int8_float16 \
  --quantization int8_float16 \
  --copy_files tokenizer.json preprocessor_config.json
```

Then load:

```python
model = WhisperModel(
    "./whisper-large-v2-ct2-int8_float16",
    device="cuda",
    compute_type="int8_float16"
)
```

### 5. Batch processing (server deployment)

```python
from faster_whisper import BatchedInferencePipeline
from pathlib import Path
from tqdm import tqdm

model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
pipeline = BatchedInferencePipeline(model=model, batch_size=8)

audio_files = list(Path("batch/").glob("*.mp3"))
for segments, info in tqdm(pipeline(audio_files), total=len(audio_files)):
    # save results...
    pass
```

Enjoy fast, memory-efficient Whisper inference on both your Mac M1 and Windows/GTX 1660 setup!