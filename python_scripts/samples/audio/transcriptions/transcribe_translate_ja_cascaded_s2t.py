from typing import Literal, Any, Dict, Union, List, Optional, Iterator, Tuple
from pathlib import Path
import time
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt
import torch
import librosa
import soundfile as sf
from transformers import pipeline
from rich.live import Live               # ← new
from rich.text import Text               # ← new
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich import print as rprint

TaskMode = Literal["transcribe", "translate"]
OutputMode = Literal["basic", "with_timestamps", "verbose"]
ComputeType = Literal["float32", "bfloat16"]

# Global cache + console
_PIPELINE_CACHE: Dict[str, Any] = {}
_console = Console()


def chunk_audio(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    max_chunk_seconds: float,
    overlap_seconds: float,
) -> List[npt.NDArray[np.float32]]:
    """
    Split 1D float32 audio into overlapping chunks (zero-copy slices).
    Same logic as used in JapaneseASR class.
    """
    if len(audio) == 0:
        return []

    chunk_size = int(max_chunk_seconds * sample_rate)
    overlap = int(overlap_seconds * sample_rate)
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError(
            f"overlap ({overlap_seconds}s) >= max_chunk_seconds ({max_chunk_seconds}s)"
        )

    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunks.append(audio[start:end])
        start += step

    return chunks


def _stitch_texts_with_overlap(prev: str, curr: str) -> str:
    """Naive but effective stitching for Japanese — looks for longest matching suffix-prefix."""
    if not prev:
        return curr
    if not curr:
        return prev

    max_match_len = 40
    prev_end = prev[-max_match_len:]
    curr_start = curr[:max_match_len]

    best_overlap = 0
    for ol in range(min(len(prev_end), len(curr_start)), 3, -1):
        if prev_end[-ol:] == curr_start[:ol]:
            best_overlap = ol
            break

    if best_overlap >= 4:
        return prev + curr[best_overlap:]
    else:
        return prev + " " + curr


def japanese_speech_to_text(
    audio_path: Union[str, Path],
    task: TaskMode = "transcribe",
    output_mode: OutputMode = "basic",
    tgt_lang: str = "eng_Latn",
    max_chunk_seconds: float = 45.0,           # ← new default (was 15 → now matches previous impl)
    chunk_overlap_seconds: float = 10.0,       # ← new: overlap for stitching quality
    device: Union[int, str, None] = None,
    compute_type: ComputeType = "float32",
    show_progress: bool = True,
) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:

    # ─── (existing body unchanged until after pipeline loading) ───

    global _PIPELINE_CACHE

    if task not in ("transcribe", "translate"):
        raise ValueError(f"Invalid task: {task!r}")
    if output_mode not in ("basic", "with_timestamps", "verbose"):
        raise ValueError(f"Invalid output_mode: {output_mode!r}")

    # ─── Device auto-detection ─────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if isinstance(device, int):
        device = f"cuda:{device}" if device >= 0 else "cpu"
    device_str = str(device)

    # ─── dtype mapping & safety fallbacks ──────────────────────────────────
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(compute_type)
    if torch_dtype is None:
        raise ValueError(f"Unsupported compute_type: {compute_type}")

    if device_str.startswith("mps") and compute_type != "float32":
        _console.print("[yellow]MPS: falling back to float32 (limited bfloat16 support)[/yellow]")
        torch_dtype = torch.float32
    elif device_str == "cpu" and compute_type != "float32":
        _console.print(f"[yellow]CPU note: {compute_type} may be slower than float32[/yellow]")

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ─── Load audio once (mono, 16kHz) ─────────────────────────────────────
    with _console.status("[cyan]Loading audio…[/cyan]"):
        array, orig_sr = librosa.load(audio_path, sr=16000, mono=True)
        duration_s = len(array) / 16000

    approx_chunks = max(1, int(duration_s / max_chunk_seconds) + 2)  # conservative estimate

    # ─── Cache key (ignores new chunk params — safe for now) ───────────────
    cache_key = f"{task}_{device_str}_{max_chunk_seconds:.1f}_{compute_type}"

    if cache_key not in _PIPELINE_CACHE:
        if show_progress:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as p:
                tid = p.add_task("[cyan]Loading model…", total=None)
                _load_pipeline(cache_key, task, tgt_lang, device_str, torch_dtype)
                p.update(tid, completed=True)
        else:
            _console.print("[cyan]Loading model…[/cyan]", end=" ")
            _load_pipeline(cache_key, task, tgt_lang, device_str, torch_dtype)
            _console.print("[green]done[/green]")

    pipe = _PIPELINE_CACHE[cache_key]

    start_time = time.time()
    full_text = ""
    chunk_results = []  # for verbose / timestamps mode

    is_long = duration_s > max_chunk_seconds + 1.0

    if not is_long:
        # ── Short audio: single pipeline call ──────────────────────────────
        chunks_info = "single pass"
        with _progress_context(show_progress, approx_chunks, duration_s) as update_fn:
            result = pipe(
                str(audio_path),
                return_timestamps=True,
                chunk_length_s=None,  # disable internal chunking
            )
            update_fn(1)

        if output_mode == "basic":
            full_text = result["text"].strip()
        else:
            chunk_results = result.get("chunks", [])
            if not chunk_results:
                chunk_results = [{"text": result["text"], "timestamp": None}]
            full_text = " ".join(c["text"] for c in chunk_results).strip()

    else:
        # ── Long audio: manual overlapping chunks + stitching ──────────────
        chunks_info = f"{approx_chunks} overlapping chunks + stitching"

        audio_chunks = chunk_audio(
            array,
            sample_rate=16000,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=chunk_overlap_seconds,
        )

        texts: List[str] = []

        with _progress_context(show_progress, len(audio_chunks), duration_s) as update_fn:
            for i, chunk in enumerate(audio_chunks, 1):
                result = pipe(
                    chunk,                               # np.float32 at 16 kHz — do NOT pass sampling_rate=
                    return_timestamps=True,
                    chunk_length_s=None,                 # disable any internal auto-chunking
                )
                text = result["text"].strip()
                texts.append(text)

                if output_mode != "basic" and "chunks" in result:
                    # Offset timestamps roughly — not perfect but better than nothing
                    offset = (i - 1) * (max_chunk_seconds - chunk_overlap_seconds)
                    for c in result["chunks"]:
                        if c.get("timestamp") and isinstance(c["timestamp"], (tuple, list)):
                            ts = c["timestamp"]
                            start = ts[0] + offset if ts[0] is not None else None
                            end   = ts[1] + offset if ts[1] is not None else None
                            c["timestamp"] = (start, end)
                        chunk_results.append(c)

                update_fn(1)

        # Stitch texts
        if texts:
            full_text = texts[0]
            for next_text in texts[1:]:
                full_text = _stitch_texts_with_overlap(full_text, next_text)
            full_text = full_text.strip()

    elapsed = time.time() - start_time

    # ─── Format output ──────────────────────────────────────────────────────
    if output_mode == "basic":
        if show_progress:
            _console.rule("Transcription / Translation")
            rprint(full_text)
            _console.rule()
            _console.print(f"[italic]Done in {elapsed:.2f}s — {chunks_info}[/italic]")
        return full_text

    elif output_mode == "with_timestamps":
        if show_progress:
            _console.rule("Chunked Result")
            for i, chunk in enumerate(chunk_results or [{"text": full_text, "timestamp": None}], 1):
                ts = chunk.get("timestamp")
                ts_str = f"{ts[0]:.1f}–{ts[1]:.1f}s" if ts and ts[0] is not None else "n/a"
                rprint(f"[dim]{i:2d}[/dim]  [blue]{ts_str}[/blue]  {chunk['text']}")
            _console.rule()
        return chunk_results or [{"text": full_text, "timestamp": None}]

    else:  # verbose
        info = {
            "text": full_text,
            "chunks_info": chunks_info,
            "duration_s": round(duration_s, 1),
            "elapsed_s": round(elapsed, 2),
        }
        if show_progress:
            table = Table(title="Verbose Result")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            for k, v in info.items():
                table.add_row(k, str(v))
            rprint(table)
        return info

def japanese_speech_to_text_stream(
    audio_path: Union[str, Path],
    task: TaskMode = "transcribe",
    tgt_lang: str = "eng_Latn",
    max_chunk_seconds: float = 45.0,
    chunk_overlap_seconds: float = 10.0,
    device: Union[int, str, None] = None,
    compute_type: ComputeType = "float32",
) -> Iterator[Tuple[str, float]]:
    """
    Streaming version: yields (current_best_text_so_far, progress_fraction) repeatedly.

    Progress ∈ [0.0 .. 1.0]. Last yield has exactly 1.0.
    Use with rich.live.Live for nice updating console UI.
    """
    global _PIPELINE_CACHE

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ─── Device & dtype logic (same as original) ───────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"

    device_str = str(device)
    if isinstance(device, int):
        device_str = f"cuda:{device}" if device >= 0 else "cpu"

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(compute_type, torch.float32)

    if "mps" in device_str and compute_type != "float32":
        _console.print("[yellow]MPS: falling back to float32[/yellow]")
        torch_dtype = torch.float32

    # ─── Load audio ────────────────────────────────────────────────────────
    with _console.status("[cyan]Loading audio…[/cyan]"):
        array, orig_sr = librosa.load(audio_path, sr=16000, mono=True)
    duration_s = len(array) / 16000.0

    # ─── Load pipeline (reuse cache logic) ─────────────────────────────────
    cache_key = f"{task}_{device_str}_{max_chunk_seconds:.1f}_{compute_type}"
    if cache_key not in _PIPELINE_CACHE:
        _load_pipeline(cache_key, task, tgt_lang, device_str, torch_dtype)
    pipe = _PIPELINE_CACHE[cache_key]

    if duration_s <= max_chunk_seconds + 1.0:
        # Short audio: single inference
        result = pipe(str(audio_path), return_timestamps=True, chunk_length_s=None)
        text = (result.get("text") or "").strip()
        yield text, 1.0
        return

    # Long audio: overlapping chunks
    chunks = chunk_audio(array, 16000, max_chunk_seconds, chunk_overlap_seconds)

    current_text = ""
    processed_up_to_s = 0.0

    for i, chunk_array in enumerate(chunks, 1):
        result = pipe(
            chunk_array,
            return_timestamps=True,
            chunk_length_s=None,
        )
        chunk_text = (result.get("text") or "").strip()

        if i == 1:
            current_text = chunk_text
        else:
            current_text = _stitch_texts_with_overlap(current_text, chunk_text)

        processed_up_to_s += max_chunk_seconds - chunk_overlap_seconds
        progress = min(1.0, processed_up_to_s / duration_s)

        yield current_text.strip(), progress

    # Final yield with clean 1.0
    yield current_text.strip(), 1.0



def _load_pipeline(
    cache_key: str,
    task: str,
    tgt_lang: str,
    device: str,
    torch_dtype: torch.dtype,
):
    common = {
        "device": device,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "model_kwargs": {"attn_implementation": "sdpa"},
    }

    if task == "transcribe":
        _PIPELINE_CACHE[cache_key] = pipeline(
            "automatic-speech-recognition",
            model="japanese-asr/ja-cascaded-s2t-translation",
            **common,
        )
    else:
        _PIPELINE_CACHE[cache_key] = pipeline(
            model="japanese-asr/ja-cascaded-s2t-translation",
            model_translation="facebook/nllb-200-distilled-600M",
            tgt_lang=tgt_lang,
            **common,
        )


@contextmanager
def _progress_context(show: bool, total: int, duration: float):
    """Helper to yield update function — simulates or uses real progress."""
    if not show:
        yield lambda _: None
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(
                f"[cyan]Processing (~{total} chunks | {duration:.1f}s)",
                total=total,
            )

            def update(adv: int = 1):
                progress.advance(task_id, advance=adv)

            yield update


# ─── Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    AUDIO_SHORT = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
    AUDIO_LONG  = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    rprint("[bold cyan]Demo: Long audio — streaming version[/bold cyan]")
    
    # ── Option A: simple print (overwrites line) ───────────────────────────
    # for text, prog in japanese_speech_to_text_stream(AUDIO_LONG):
    #     print(f"\r[{prog:5.1%}] {text}", end="", flush=True)
    # print()

    # ── Option B: nice updating UI with rich ───────────────────────────────
    with Live(refresh_per_second=4, console=_console) as live:
        last_text = ""
        for text, progress in japanese_speech_to_text_stream(
            AUDIO_LONG,
            task="transcribe",
            # compute_type="bfloat16",  # if your GPU likes it
        ):
            if text != last_text or progress >= 1.0:
                live.update(
                    Text.from_markup(f"[cyan]{progress:>6.1%}[/cyan]  {text}")
                )
                last_text = text

    rprint("[green]Streaming transcription complete.[/green]")

    # ── Original non-streaming call still works ────────────────────────────
    # rprint("\n[bold]Original non-streaming version:[/bold]")
    # japanese_speech_to_text(
    #     AUDIO_LONG,
    #     task="transcribe",
    #     output_mode="basic",
    #     # compute_type="bfloat16",
    # )