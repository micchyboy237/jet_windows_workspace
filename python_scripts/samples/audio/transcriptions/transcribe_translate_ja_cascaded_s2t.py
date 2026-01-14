from typing import Literal, Any, Dict, Union, List, Optional
from pathlib import Path
import time

import torch
import soundfile as sf
from transformers import pipeline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich import print as rprint

TaskMode = Literal["transcribe", "translate"]
OutputMode = Literal["basic", "with_timestamps", "verbose"]
ComputeType = Literal["float32", "float16", "bfloat16"]

# Global cache + console
_PIPELINE_CACHE: Dict[str, Any] = {}
_console = Console()


def japanese_speech_to_text(
    audio_path: Union[str, Path],
    task: TaskMode = "transcribe",
    output_mode: OutputMode = "basic",
    tgt_lang: str = "eng_Latn",
    chunk_length_s: float = 15.0,
    device: Union[int, str, None] = None,   # None → auto-detect
    compute_type: ComputeType = "float32",
    show_progress: bool = True,
) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Reusable Japanese speech → text / translation function with rich progress & logging.

    Progress estimation:
    - Uses real audio duration + chunk_length_s to show approximate chunk progress
    - Rich spinner + bar during inference
    - Clean logging sections (loading, audio info, result summary)

    New parameters:
        device:        "cuda", "cuda:0", "mps", "cpu", 0, -1, None (auto)
        compute_type:  "float32" (default), "float16", "bfloat16"
                       → sets torch_dtype for faster / lower-memory inference
        show_progress: bool = True   → toggle rich progress UI (useful for notebooks/scripts)
    """
    global _PIPELINE_CACHE

    if task not in ("transcribe", "translate"):
        raise ValueError(f"Invalid task: {task!r}")
    if output_mode not in ("basic", "with_timestamps", "verbose"):
        raise ValueError(f"Invalid output_mode: {output_mode!r}")

    # ─── Auto-detect device if None ────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Normalize device string
    if isinstance(device, int):
        device = f"cuda:{device}" if device >= 0 else "cpu"
    device_str = str(device)

    # ─── Map compute_type string → torch.dtype ──────────────────────────────
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(compute_type)
    if torch_dtype is None:
        raise ValueError(f"Unsupported compute_type: {compute_type}")

    # Safety fallback / warning for problematic combinations
    if device_str.startswith("mps") and compute_type != "float32":
        _console.print(
            "[yellow]Warning: MPS has limited / unstable support for "
            f"{compute_type}. Falling back to float32.[/yellow]"
        )
        torch_dtype = torch.float32
    elif device_str == "cpu" and compute_type != "float32":
        _console.print(
            f"[yellow]Note: Using {compute_type} on CPU — may be slower than float32.[/yellow]"
        )

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cache_key = f"{task}_{device_str}_{chunk_length_s:.1f}_{compute_type}"

    # ─── Model loading with progress ────────────────────────────────────────
    if cache_key not in _PIPELINE_CACHE:
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task_id = progress.add_task("[cyan]Loading model…", total=None)
                _load_pipeline(
                    cache_key,
                    task,
                    tgt_lang,
                    chunk_length_s,
                    device_str,
                    torch_dtype,
                )
                progress.update(task_id, completed=True)
        else:
            _console.print("[cyan]Loading model…[/cyan]", end=" ")
            _load_pipeline(cache_key, task, tgt_lang, chunk_length_s, device_str, torch_dtype)
            _console.print("[green]done[/green]")

    pipe = _PIPELINE_CACHE[cache_key]

    # ─── Get real audio duration for better progress estimation ─────────────
    with sf.SoundFile(str(audio_path)) as f:
        if f.samplerate <= 0:
            raise ValueError("Invalid sample rate detected")
        duration_s = f.frames / f.samplerate
        approx_chunks = max(1, int(duration_s / chunk_length_s) + 1)

    # ─── Inference with rich progress ───────────────────────────────────────
    start_time = time.time()

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"[cyan]Processing audio (~{approx_chunks} chunks)" + (f" | {duration_s:.1f}s" if duration_s else ""),
                total=approx_chunks,
            )

            # We can't get real per-chunk callbacks → simulate progress
            # Advance ~evenly over estimated chunks
            result = pipe(
                str(audio_path),
                return_timestamps=(output_mode == "with_timestamps"),
            )
            # Fake progress completion (since no real callback)
            progress.update(task_id, completed=approx_chunks)
    else:
        result = pipe(
            str(audio_path),
            return_timestamps=(output_mode == "with_timestamps"),
        )

    elapsed = time.time() - start_time

    # ─── Format output ──────────────────────────────────────────────────────
    if output_mode == "basic":
        text = result["text"]
        if show_progress:
            _console.rule("Result")
            rprint(text)
            _console.rule()
        return text

    elif output_mode == "with_timestamps":
        chunks = result.get("chunks", [])
        if not chunks:
            chunks = [{"text": result["text"], "timestamp": None}]
        if show_progress:
            _console.rule("Chunked Result")
            for i, chunk in enumerate(chunks, 1):
                ts = chunk.get("timestamp")
                ts_str = f"{ts[0]:.1f}–{ts[1]:.1f}s" if isinstance(ts, (tuple, list)) and ts[0] is not None else "n/a"
                rprint(f"[dim]{i:2d}[/dim]  [blue]{ts_str}[/blue]  {chunk['text']}")
            _console.rule()
        return chunks

    else:  # verbose
        if show_progress:
            table = Table(title="Verbose Result", show_header=True, header_style="bold magenta")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            for k, v in result.items():
                if k == "chunks" and isinstance(v, list):
                    table.add_row(k, f"{len(v)} chunks")
                else:
                    table.add_row(k, str(v)[:120] + "..." if len(str(v)) > 120 else str(v))
            rprint(table)
            _console.print(f"[italic]Processed in {elapsed:.2f} seconds[/italic]")
        return result


def _load_pipeline(
    cache_key: str,
    task: str,
    tgt_lang: str,
    chunk_length_s: float,
    device: str,
    torch_dtype: torch.dtype,
):
    common_kwargs = {
        "chunk_length_s": chunk_length_s,
        "device": device,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if task == "transcribe":
        _PIPELINE_CACHE[cache_key] = pipeline(
            "automatic-speech-recognition",
            model="japanese-asr/ja-cascaded-s2t-translation",
            model_kwargs={"attn_implementation": "sdpa"},
            **common_kwargs,
        )
    else:
        _PIPELINE_CACHE[cache_key] = pipeline(
            model="japanese-asr/ja-cascaded-s2t-translation",
            model_kwargs={"attn_implementation": "sdpa"},
            model_translation="facebook/nllb-200-distilled-600M",
            tgt_lang=tgt_lang,
            **common_kwargs,
        )


# ─── Updated usage examples ─────────────────────────────────────────────────

if __name__ == "__main__":
    AUDIO_SHORT = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
    AUDIO_LONG  = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    # Most common cases
    rprint("[bold cyan]Demo: auto device + float32[/bold cyan]")
    japanese_speech_to_text(AUDIO_LONG)                                 # auto device + float32
    # rprint("[bold cyan]Demo: try half precision[/bold cyan]")
    # japanese_speech_to_text(AUDIO_SHORT, compute_type="float16")         # try half precision
    # rprint("[bold cyan]Demo: long audio with half precision[/bold cyan]")
    # japanese_speech_to_text(AUDIO_LONG, compute_type="float16")
