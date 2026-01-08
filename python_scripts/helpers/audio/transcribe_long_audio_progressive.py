from typing import Generator, Optional, Tuple
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.text import Text
from tqdm import tqdm

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

console = Console()


def transcribe_progressive(
    model: WhisperModel,
    audio: np.ndarray,
    sr: int = 16000,
    language: Optional[str] = None,
    task: str = "transcribe",
    vad_filter: bool = True,
    condition_on_previous_text: bool = True,
    beam_size: int = 5,
    word_timestamps: bool = False,
    show_progress: bool = False,
    **transcribe_kwargs,
) -> Generator[Tuple[Segment, str], None, None]:
    """
    Transcribes audio and yields each new/changed segment immediately for UI updates.
    Uses rich Live display as example — replace with your own UI (Streamlit, gradio, console, websocket...).

    Yields: (latest_segment, current_full_transcript_text)
    """
    full_text_parts = []
    last_printed_length = 0

    # Optional: show simple progress bar while waiting for first segments
    with tqdm(desc="Transcribing...", unit="seg", disable=not show_progress) as pbar:

        segments, info = model.transcribe(
            audio,
            language=language,
            task=task,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            **transcribe_kwargs,
        )

        for segment in segments:
            pbar.update(1)

            # Build running text (you can also use segment.text directly)
            full_text_parts.append(segment.text.strip())
            current_text = " ".join(full_text_parts).strip()

            # Yield the new segment + full current transcription
            yield segment, current_text

            # Live console preview example
            if len(current_text) > last_printed_length:
                console.print(
                    Text.from_markup(
                        f"[dim]{segment.start:5.1f}s → {segment.end:5.1f}s[/dim]  "
                        f"[bold]{segment.text.strip()}[/bold]"
                    )
                )
                last_printed_length = len(current_text)


# ──────────────────────────────────────────────────────────────────────────────
#   Usage example
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server_spyxfamily_intro\utterance_1d825427_0001_20260104_061420.wav"

    # Load model once (reuse for multiple calls)
    model = WhisperModel(
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        device="cuda",
        compute_type="float32",
    )

    # Example: load your 20-second (or longer) audio
    from scipy.io import wavfile
    sr, audio = wavfile.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)          # to mono
    audio = audio.astype(np.float32) / 32768.0

    # Real-time-like progressive display
    with Live(console=console, refresh_per_second=4) as live:
        full_transcript = ""

        for segment, current_text in transcribe_progressive(
            model,
            audio,
            sr=sr,
            language="ja",
            beam_size=5,
            show_progress=True,
        ):
            full_transcript = current_text
            # Update live display (replace with your frontend)
            live.update(
                f"[bold green]Live transcript:[/bold green]\n{full_transcript}\n\n"
                f"[dim]Latest segment: {segment.start:.1f}–{segment.end:.1f}s[/dim]"
            )

    console.rule("Final result")
    console.print(full_transcript)