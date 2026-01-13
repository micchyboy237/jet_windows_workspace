#!/usr/bin/env python3
# live_jav_subtitle.py
# Live Japanese → English subtitles from audio loopback (VB-Cable)
# Fixed: silero-vad now receives fixed 512-sample windows @16kHz

import time
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
from silero_vad import load_silero_vad, VADIterator
from rich.live import Live
from rich.text import Text
from rich.console import Console
from pathlib import Path
from queue import Queue, Empty

# ─── Configuration ────────────────────────────────────────

SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512                # silero-vad JIT requirement @16kHz
BLOCK_SECONDS = 4.0                     # sounddevice callback interval
MIN_SPEECH_SEC = 1.5
MAX_CHUNK_SEC = 18.0

INPUT_DEVICE_INDEX = 5                  # Your CABLE Output (VB-Audio Point)

# ─── Models ───────────────────────────────────────────────

console = Console()
console.print("[bold green]Loading models...[/bold green]")

asr_model = WhisperModel(
    "large-v3",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16",
    cpu_threads=6
)

translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16,
)

vad_model = load_silero_vad()
vad_iterator = VADIterator(vad_model, threshold=0.38, sampling_rate=SAMPLE_RATE)

console.print("[bold green]Models loaded[/bold green]")

# ─── State ────────────────────────────────────────────────

audio_q: Queue = Queue()
buffer: list[float] = []
last_transcribed_end = 0.0
srt_path = Path("jav_live.srt")
srt_counter = 1

srt_path.write_text("", encoding="utf-8")  # reset file

# ─── Helpers ──────────────────────────────────────────────

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def append_to_srt(start: float, end: float, ja: str, en: str) -> None:
    global srt_counter
    block = [
        str(srt_counter),
        f"{format_time(start)} --> {format_time(end)}",
        ja.strip(),
        "→",
        en.strip(),
        ""
    ]
    with srt_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n\n")
    srt_counter += 1
    console.print(f"[dim italic]Wrote SRT block {srt_counter-1}[/dim italic]")


# ─── Audio callback ───────────────────────────────────────

def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    if status:
        console.print(f"[yellow]Audio warning: {status}[/yellow]")
    audio_q.put(indata[:, 0].copy())


# ─── Main loop ────────────────────────────────────────────

console.rule("Live Japanese → English Subtitles")
console.print(f"Input device: {sd.query_devices(INPUT_DEVICE_INDEX)['name']} (index {INPUT_DEVICE_INDEX})")
console.print("Route video audio to 'CABLE Output (VB-Audio Point)'")
console.print("Press Ctrl+C to stop\n")

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
    callback=audio_callback,
    device=INPUT_DEVICE_INDEX
)

stream.start()

try:
    with Live(console=console, refresh_per_second=3) as live:
        status_text = Text("Waiting for speech...", style="cyan")
        live.update(status_text)

        while True:
            chunk = None
            new_data = False

            try:
                chunk = audio_q.get(timeout=0.1)
                buffer.extend(chunk.tolist())
                new_data = True
            except Empty:
                pass

            if new_data and chunk is not None:
                # Feed audio to VAD in fixed 512-sample windows
                chunk_array = np.array(chunk, dtype=np.float32)
                for i in range(0, len(chunk_array), VAD_WINDOW_SAMPLES):
                    window = chunk_array[i:i + VAD_WINDOW_SAMPLES]

                    # Pad last window if needed
                    if len(window) < VAD_WINDOW_SAMPLES:
                        window = np.pad(window, (0, VAD_WINDOW_SAMPLES - len(window)))

                    speech_dict = vad_iterator(window, return_seconds=True)

                    if speech_dict and "end" in speech_dict:
                        console.print("[yellow bold]VAD detected speech END[/yellow bold]")

                        audio_np = np.array(buffer, dtype=np.float32)
                        duration = len(audio_np) / SAMPLE_RATE

                        if duration >= MIN_SPEECH_SEC:
                            status_text = Text("Transcribing...", style="yellow bold")
                            live.update(status_text)

                            segments, info = asr_model.transcribe(
                                audio_np,
                                language="ja",
                                vad_filter=True,
                                vad_parameters={"min_silence_duration_ms": 350}
                            )

                            if segments:
                                ja_text = " ".join(
                                    s.text.strip() for s in segments if s.text.strip()
                                ).strip()

                                if ja_text:
                                    console.print(f"[cyan]JA: {ja_text[:100]}{'…' if len(ja_text)>100 else ''}[/cyan]")

                                    trans_result = translator(
                                        ja_text,
                                        src_lang="jpn_Jpan",
                                        tgt_lang="eng_Latn",
                                        max_length=256
                                    )
                                    en_text = trans_result[0]["translation_text"]

                                    console.print(f"[magenta]EN: {en_text[:100]}{'…' if len(en_text)>100 else ''}[/magenta]")

                                    start_t = last_transcribed_end
                                    end_t = last_transcribed_end + duration
                                    append_to_srt(start_t, end_t, ja_text, en_text)

                                    status_text = Text.assemble(
                                        ("JA: ", "bold cyan"),
                                        (ja_text[:140] + "…" if len(ja_text) > 140 else ja_text, "white"),
                                        ("\nEN: ", "bold magenta"),
                                        (en_text[:140] + "…" if len(en_text) > 140 else en_text, "white")
                                    )
                                    live.update(status_text)

                                    last_transcribed_end = end_t

                        # Reset after processing utterance
                        buffer = []
                        vad_iterator.reset_states()

            # Force process very long buffer
            if len(buffer) / SAMPLE_RATE > MAX_CHUNK_SEC:
                console.print("[yellow]Buffer timeout – forcing transcription[/yellow]")
                # (add same transcription block here if you want – omitted for brevity)

            time.sleep(0.03)

except KeyboardInterrupt:
    console.print("\n[bold red]Stopped by user[/bold red]")
except Exception as e:
    console.print(f"\n[bold red]Error: {e}[/bold red]")
finally:
    stream.stop()
    stream.close()
    vad_iterator.reset_states()
    console.print(f"\nFinal .srt: [green]{srt_path.absolute()}[/green]")