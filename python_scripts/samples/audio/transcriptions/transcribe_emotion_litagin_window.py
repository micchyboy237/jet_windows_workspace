from transformers import pipeline
import numpy as np
from scipy.io import wavfile
import os
from rich import print as rprint
from rich.table import Table

# ──── Config ───────────────────────────────────────────────────────
WINDOW_SEC = 6.0
STEP_SEC   = 2.0
audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

# ──── Load audio once ──────────────────────────────────────────────
sr, data = wavfile.read(audio_path)
if len(data.shape) == 2:           # stereo → mono
    data = data.mean(axis=1).astype(data.dtype)
duration = len(data) / sr

# ──── Prepare pipeline ─────────────────────────────────────────────
pipe = pipeline(
    "audio-classification",
    model="litagin/anime_speech_emotion_classification",
    feature_extractor="litagin/anime_speech_emotion_classification",
    trust_remote_code=True,
    device="cuda",
)

# ──── Process sliding windows ──────────────────────────────────────
results = []
times = []  # middle time of each window

t_start = 0.0
while t_start + WINDOW_SEC <= duration:
    t_end = t_start + WINDOW_SEC
    start_sample = int(t_start * sr)
    end_sample   = int(t_end   * sr)

    chunk = data[start_sample:end_sample]

    # temporary wav for pipeline
    tmp_path = "tmp_chunk.wav"
    wavfile.write(tmp_path, sr, chunk)

    pred = pipe(tmp_path)[0]           # top result with score & label
    os.remove(tmp_path)

    mid_time = t_start + WINDOW_SEC / 2
    results.append(pred)
    times.append(mid_time)

    t_start += STEP_SEC

# ──── Pretty table output ──────────────────────────────────────────
table = Table(title="Sliding Window Emotion Predictions")
table.add_column("Time (middle)", justify="right")
table.add_column("Top Emotion")
table.add_column("Confidence", justify="right")

for t, r in zip(times, results):
    table.add_row(f"{t:5.1f}s", r['label'], f"{r['score']:.1%}")

rprint(table)

# Bonus: also show top-3 for each window if you want