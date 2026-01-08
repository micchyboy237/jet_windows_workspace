"""
anime_whisper_demos.py
Use-case demos for litagin/anime-whisper
"""

from __future__ import annotations

import csv
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import pipeline
from tqdm import tqdm


# -----------------------------------------------------------
# Initialize one shared pipeline
# -----------------------------------------------------------
ASR = pipeline(
    "automatic-speech-recognition",
    model="litagin/anime-whisper",
    device="cuda",
    torch_dtype=torch.float16,
    chunk_length_s=30.0,
)


# -----------------------------------------------------------
# 🎭 1) Anime-style transcription
# -----------------------------------------------------------
def demo_anime_style(audio_file: str) -> str:
    result = ASR(audio_file)
    text = result["text"]
    print(text)
    return text


# -----------------------------------------------------------
# 🎮 2) Visual novel bulk extraction
# -----------------------------------------------------------
def demo_visual_novel(folder: str) -> List[Tuple[str, str]]:
    audio_files = list(Path(folder).glob("**/*.wav"))
    lines: List[Tuple[str, str]] = []
    for file in tqdm(audio_files, desc="Transcribing VN voices"):
        out = ASR(str(file))
        lines.append((file.name, out["text"]))
    for fn, text in lines[:10]:
        print(fn, "→", text)
    return lines


# -----------------------------------------------------------
# 🎙 3) Voice actor timestamp logging
# -----------------------------------------------------------
def demo_voice_actor(audio_file: str, log_file: str = "session_log.txt") -> str:
    text = ASR(audio_file)["text"]
    with open(log_file, "a", encoding="utf8") as f:
        f.write(f"{datetime.now()} :: {audio_file} :: {text}\n")
    print(text)
    return text


# -----------------------------------------------------------
# 📺 4) Subtitle draft (simple SRT-like output)
# -----------------------------------------------------------
def demo_subtitle_draft(audio_file: str, out_file: str = "draft.srt") -> None:
    text = ASR(audio_file)["text"]
    with open(out_file, "w", encoding="utf8") as f:
        for i, line in zip(count(1), text.split("、")):
            f.write(f"{i}\n00:00:00,000 --> 00:00:05,000\n{line.strip()}\n\n")
    print(f"Draft written to {out_file}")


# -----------------------------------------------------------
# 😂 5) Emotional / nonverbal capture
# -----------------------------------------------------------
def demo_emotional(audio_file: str) -> str:
    result = ASR(audio_file, generate_kwargs={"no_repeat_ngram_size": 0})
    text = result["text"]
    print(text)
    return text


# -----------------------------------------------------------
# 🔞 6) NSFW transcription
# -----------------------------------------------------------
def demo_nsfw(audio_file: str) -> str:
    text = ASR(audio_file)["text"]
    print(text)
    return text


# -----------------------------------------------------------
# 🧪 7) Dataset labeling to CSV
# -----------------------------------------------------------
def demo_dataset(folder: str, csv_out: str = "labels.csv") -> None:
    files = list(Path(folder).glob("*.wav"))
    with open(csv_out, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "transcript"])
        for file in files:
            txt = ASR(str(file))["text"]
            writer.writerow([file.name, txt])
    print(f"Wrote {len(files)} rows → {csv_out}")


# -----------------------------------------------------------
# 📡 8) VTuber / livestream clipping
# -----------------------------------------------------------
def demo_stream_clips(folder: str) -> None:
    for wav in Path(folder).glob("*.wav"):
        print(wav.name, "→", ASR(str(wav))["text"])


# -----------------------------------------------------------
# Runner (optional manual testing)
# -----------------------------------------------------------
if __name__ == "__main__":
    # Uncomment and point to files to try individual demos:
    # demo_anime_style("voice.wav")
    # demo_visual_novel("voices/")
    # demo_voice_actor("take_01.wav")
    # demo_subtitle_draft("scene.wav")
    # demo_emotional("reaction.wav")
    # demo_nsfw("adult_clip.wav")
    # demo_dataset("dataset/")
    # demo_stream_clips("clips/")
    pass
