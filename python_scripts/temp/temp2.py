from typing import List, Optional, Iterable
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.logging import RichHandler
import logging
import time
from dataclasses import dataclass
from pathlib import Path

# pydub for audio loading & splitting
try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence  # optional
except ImportError:
    print("pydub not installed → run:  pip install pydub")
    print("Also need ffmpeg installed on your system")
    raise

# ────────────────────────────────────────────────
#          Logging Setup
# ────────────────────────────────────────────────

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("JA→EN")


@dataclass
class ChunkStats:
    chunk_idx: int
    chunk_duration_sec: float
    num_segments: int
    text_length: int
    context_length_before: int
    context_length_after: int
    processing_time_sec: float


class ContextualJapaneseToEnglishTranscriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        context_window_chars: int = 450,
        prefix_chars: int = 140,
    ):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.console = console
        self.previous_english: str = ""
        self.context_window_chars = context_window_chars
        self.prefix_chars = prefix_chars
        self.stats: List[ChunkStats] = []

    def _build_prompt(self) -> str:
        if not self.previous_english:
            return (
                "Translate Japanese speech to natural, fluent, conversational English. "
                "Preserve speaker personality, names, technical terms, humor, cultural references, "
                "and full context across sentences."
            )

        context = (
            self.previous_english[-self.context_window_chars:]
            if len(self.previous_english) > self.context_window_chars
            else self.previous_english
        )
        return (
            "Translate Japanese speech to natural, fluent, conversational English. "
            "Preserve speaker personality, names, technical terms, humor, cultural references, "
            "and full context across sentences.\n"
            f"Previous English context:\n{context}"
        )

    def transcribe_chunk(
        self,
        audio_chunk: np.ndarray,
        chunk_idx: int,
        total_chunks: int,
        hotwords: Optional[str] = None,
    ) -> List[Segment]:
        start_time = time.perf_counter()

        prompt = self._build_prompt()
        prefix = self.previous_english[-self.prefix_chars:] if self.previous_english else None

        logger.info(
            f"[bold cyan]Chunk {chunk_idx+1}/{total_chunks}[/bold cyan]  "
            f"context={len(self.previous_english):,} chars   "
            f"audio={audio_chunk.shape[0]/16000:.1f}s"
        )

        segments, info = self.model.transcribe(
            audio_chunk,
            language="ja",
            task="translate",
            initial_prompt=prompt,
            prefix=prefix,
            hotwords=hotwords,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=350,
                max_speech_duration_s=22,
            ),
            beam_size=5,
            best_of=5,
            temperature=0.0,
        )

        segments_list = list(segments)
        new_text = " ".join(s.text.strip() for s in segments_list if s.text.strip())

        duration = audio_chunk.shape[0] / 16000.0
        proc_time = time.perf_counter() - start_time

        self.stats.append(
            ChunkStats(
                chunk_idx=chunk_idx,
                chunk_duration_sec=round(duration, 2),
                num_segments=len(segments_list),
                text_length=len(new_text),
                context_length_before=len(self.previous_english),
                context_length_after=len(self.previous_english) + len(new_text),
                processing_time_sec=round(proc_time, 2),
            )
        )

        if new_text:
            self.previous_english = (self.previous_english + " " + new_text).strip()

        return segments_list

    def process_chunks(
        self,
        chunks: List[np.ndarray],
        hotwords: Optional[str] = None,
    ) -> List[Segment]:
        self.stats.clear()
        all_segments: List[Segment] = []

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

        with progress:
            task = progress.add_task(
                "[cyan]Transcribing Japanese → English with context…",
                total=len(chunks)
            )

            for i, chunk in enumerate(chunks):
                segs = self.transcribe_chunk(
                    chunk,
                    chunk_idx=i,
                    total_chunks=len(chunks),
                    hotwords=hotwords,
                )
                all_segments.extend(segs)
                progress.advance(task)

        self._print_summary()
        return all_segments

    def _print_summary(self):
        if not self.stats:
            return

        table = Table(title="Transcription Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Chunk", justify="right")
        table.add_column("Duration", justify="right")
        table.add_column("Segments", justify="right")
        table.add_column("Text chars", justify="right")
        table.add_column("Context before", justify="right")
        table.add_column("Context after", justify="right")
        table.add_column("Time (s)", justify="right")

        total_duration = sum(s.chunk_duration_sec for s in self.stats)
        total_time = sum(s.processing_time_sec for s in self.stats)
        total_segments = sum(s.num_segments for s in self.stats)

        for s in self.stats:
            table.add_row(
                f"{s.chunk_idx+1}",
                f"{s.chunk_duration_sec:.1f}s",
                str(s.num_segments),
                f"{s.text_length:,}" if s.text_length else "—",
                f"{s.context_length_before:,}",
                f"{s.context_length_after:,}",
                f"{s.processing_time_sec:.2f}",
            )

        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_duration:.1f}s[/bold]",
            f"[bold]{total_segments}[/bold]",
            "",
            "",
            "",
            f"[bold]{total_time:.2f}s[/bold]",
        )

        self.console.print(table)
        self.console.print(f"\n[green]→ Final English context length: {len(self.previous_english):,} chars[/green]\n")

    def reset_context(self) -> None:
        self.previous_english = ""
        self.stats.clear()
        logger.info("[yellow]Context and stats reset[/yellow]")


# ────────────────────────────────────────────────
#          Audio Loading & Chunking
# ────────────────────────────────────────────────

def load_and_split_audio(
    audio_path: str | Path,
    chunk_length_ms: int = 28000,   # ~28 seconds — good balance for large-v3
    overlap_ms: int = 5000,         # 5s overlap helps context continuity
    target_sr: int = 16000,
) -> List[np.ndarray]:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logger.info(f"[bold green]Loading audio:[/bold green] {path.name}")

    audio = AudioSegment.from_file(str(path))
    logger.info(f"Original duration: {len(audio)/1000:.1f}s | channels: {audio.channels} | "
                f"sample rate: {audio.frame_rate} Hz")

    # Convert to mono + 16kHz
    if audio.channels > 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)

    logger.info(f"→ Converted to mono {target_sr} Hz")

    chunks = []
    step_ms = chunk_length_ms - overlap_ms
    total_length_ms = len(audio)

    for start_ms in range(0, total_length_ms, step_ms):
        end_ms = min(start_ms + chunk_length_ms, total_length_ms)
        chunk_audio = audio[start_ms:end_ms]

        # Convert to numpy float32 [-1,1]
        samples = np.array(chunk_audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0  # int16 → float32 [-1,1]

        chunks.append(samples)

    logger.info(f"→ Created {len(chunks)} overlapping chunks "
                f"({chunk_length_ms/1000:.1f}s each, {overlap_ms/1000:.1f}s overlap)")
    return chunks


# ────────────────────────────────────────────────
#          Main
# ────────────────────────────────────────────────

def main():
    console.rule("Japanese → English Contextual Transcriber")

    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"

    transcriber = ContextualJapaneseToEnglishTranscriber(
        model_size="large-v3",
        device="mps",               # ← change to "cuda" or "cpu" depending on your machine
        compute_type="float16",     # or "int8" / "int8_float16" if memory is tight
        context_window_chars=500,
        prefix_chars=160,
    )

    hotwords = (
        "Spy × Family Anya Yor Loid Twilight Thorn Princess Bond Handler Franky "
        "スパイファミリー アーニャ ヨル ロイド フォージャー ダミアン ベッキー"
    )

    try:
        chunks = load_and_split_audio(
            audio_path,
            chunk_length_ms=28000,
            overlap_ms=5000,
        )
    except Exception as e:
        console.print_exception()
        return

    logger.info("[bold green]Starting transcription...[/bold green]\n")

    start_total = time.perf_counter()

    all_segments = transcriber.process_chunks(
        chunks=chunks,
        hotwords=hotwords,
    )

    total_time = time.perf_counter() - start_total

    # ── Show results ────────────────────────────────────────
    console.print("\n[bold underline]Translated Segments (first 8 shown):[/bold underline]\n")

    for i, seg in enumerate(all_segments[:8], 1):
        console.print(f"[grey50]{seg.start:.1f}s → {seg.end:.1f}s[/grey50]   {seg.text.strip()}")

    if len(all_segments) > 8:
        console.print(f"\n… {len(all_segments)-8} more segments …")

    console.print(f"\n[green bold]Completed in {total_time:.1f} seconds[/green bold]")

    # Optional: save full transcription
    output_path = Path(audio_path).with_suffix(".en.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text.strip()}\n")
    console.print(f"[green]Full transcription saved to:[/green] {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception:
        console.print_exception(show_locals=True)