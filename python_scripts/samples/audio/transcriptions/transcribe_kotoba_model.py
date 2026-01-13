import torch
from transformers import pipeline
from typing import List, Dict, Any

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich import print as rprint

# Global rich console for consistent output
console = Console()

# Model configuration (shared across all functions)
# MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0"
MODEL_ID = "kotoba-tech/kotoba-whisper-v2.2"
GENERATE_KWARGS = {"language": "ja", "task": "transcribe"}


def create_pipeline() -> Any:
    """
    Create and return the Whisper pipeline with optimal settings for available hardware.
    
    Returns:
        The initialized transformers pipeline for automatic-speech-recognition.
    """
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
    
    console.rule("[bold blue]Initializing Kotoba-Whisper Pipeline[/bold blue]")
    rprint(f"[yellow]Using device:[/yellow] {device} | [yellow]dtype:[/yellow] {torch_dtype}")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs,
    )
    console.print("[green]✓ Pipeline created successfully[/green]")
    return pipe


def transcribe_basic(audio_path: str, pipe: Any) -> str:
    """
    Perform basic transcription on a single audio file.
    
    Args:
        audio_path: Path to the WAV file.
        pipe: Initialized transformers pipeline.
    
    Returns:
        Transcribed text as string.
    """
    console.rule("[bold magenta]Basic Transcription[/bold magenta]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Transcribing...", total=None)  # indeterminate spinner
        # The crucial fix: pass return_timestamps=True so both short and long audio work!
        result = pipe(audio_path, return_timestamps=True, generate_kwargs=GENERATE_KWARGS)
        progress.update(task, completed=100)
    
    console.print("[bold green]Transcription complete[/bold green]")
    # Handles both classic and long-form output format.
    # If "text" is present, prefer it (single string mode, e.g. for short or plain output)
    if "text" in result:
        return result["text"]
    # Fallback – just in case long-form returns a list of chunks for some models
    if "chunks" in result and isinstance(result["chunks"], list):
        return "".join(chunk["text"] for chunk in result["chunks"])
    return str(result)  # fallback for odd edge cases


def transcribe_with_timestamps(audio_path: str, pipe: Any) -> List[Dict[str, Any]]:
    """
    Perform transcription with segment-level timestamps (chunks).
    
    Args:
        audio_path: Path to the WAV file.
        pipe: Initialized transformers pipeline.
    
    Returns:
        List of chunk dictionaries containing text, timestamp, etc.
    """
    console.rule("[bold magenta]Transcription with Timestamps[/bold magenta]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Transcribing with timestamps...", total=None)
        # You may optionally add chunk_length_s=30 and batch_size=8 for very long files/VRAM issues
        result = pipe(
            audio_path,
            return_timestamps=True,
            # chunk_length_s=30,           # uncomment if you want classic fixed length
            # batch_size=8,                # uncomment for longer files on GPU
            generate_kwargs=GENERATE_KWARGS,
        )
        progress.update(task, completed=100)
    
    console.print("[bold green]Timestamped transcription complete[/bold green]")
    # Defensive: gracefully handle either direct "chunks" or a flat "segments" structure
    if "chunks" in result and isinstance(result["chunks"], list):
        return result["chunks"]
    if "segments" in result and isinstance(result["segments"], list):
        return result["segments"]
    raise RuntimeError("Could not extract timestamped chunks from result.")


def transcribe_long_form(audio_path: str, pipe: Any, chunk_length_s: int = 15) -> str:
    """
    Perform transcription suitable for longer audio files using chunking or sequential long-form decoding.
    
    Args:
        audio_path: Path to the WAV file.
        pipe: Initialized transformers pipeline.
        chunk_length_s: IGNORED! Preserved for backward compat only; now uses sequential mode.
    
    Returns:
        Full transcribed text.
    """
    # Modern pattern: chunk_length_s = 0 is now STRONGLY RECOMMENDED for quality on long audio
    console.rule(f"[bold magenta]Long-form Transcription (modern sequential)[/bold magenta]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing long audio in chunks...", total=None)
        result = pipe(
            audio_path,
            chunk_length_s=0,               # 0 = sequential/OpenAI-like (preferred from transformers 4.39+)
            return_timestamps=True,         # Required: enables segments for long-form
            generate_kwargs=GENERATE_KWARGS,
        )
        progress.update(task, completed=100)
    
    console.print("[bold green]Long-form transcription complete[/bold green]")
    # Usually returns "chunks" or "segments" (list of {text,timestamps}), or flat text
    if "text" in result:
        return result["text"]
    if "chunks" in result and isinstance(result["chunks"], list):
        return "".join(chunk["text"] for chunk in result["chunks"])
    if "segments" in result and isinstance(result["segments"], list):
        return "".join(seg["text"] for seg in result["segments"])
    return str(result)  # fallback


def transcribe_batch(audio_paths: List[str], pipe: Any) -> List[str]:
    """
    Perform batch transcription on multiple audio files.
    
    Args:
        audio_paths: List of paths to WAV files.
        pipe: Initialized transformers pipeline.
    
    Returns:
        List of transcribed texts in the same order.
    """
    console.rule(f"[bold magenta]Batch Transcription ({len(audio_paths)} file(s))[/bold magenta]")
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Transcribing batch...", total=None)
        # Add return_timestamps=True here too: ensures consistency for long files in batch!
        results = pipe(audio_paths, return_timestamps=True, generate_kwargs=GENERATE_KWARGS)
        progress.update(task, completed=100)
    
    console.print("[bold green]Batch transcription complete[/bold green]")
    # Defensive: handle result format being a list of dicts with either "text" or "chunks"
    texts = []
    for r in results:
        if isinstance(r, dict):
            if "text" in r:
                texts.append(r["text"])
            elif "chunks" in r and isinstance(r["chunks"], list):
                texts.append("".join(chunk["text"] for chunk in r["chunks"]))
            elif "segments" in r and isinstance(r["segments"], list):
                texts.append("".join(seg["text"] for seg in r["segments"]))
            else:
                texts.append(str(r))
        else:
            texts.append(str(r))
    return texts


if __name__ == "__main__":
    # Single audio file to test all examples
    SHORT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
    LONG_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    audio_path = SHORT_AUDIO
    # audio_path = LONG_AUDIO

    # Create pipeline once (model download happens here on first run)
    whisper_pipe = create_pipeline()

    console.rule("[bold]Demo: Kotoba-Whisper v2.* on Local Audio[/bold]")
    text_basic = transcribe_basic(audio_path, whisper_pipe)
    rprint(f"[white]{text_basic}[/white]")
    console.print()

    console.rule("[bold]Transcription with Timestamps[/bold]")
    chunks = transcribe_with_timestamps(audio_path, whisper_pipe)
    for chunk in chunks:
        start, end = chunk.get("timestamp", (None, None))
        rprint(f"[yellow][{start:.2f}s → {end:.2f}s][/yellow] {chunk.get('text','')}")
    console.print()

    console.rule("[bold]Long-form Transcription[/bold]")
    text_long = transcribe_long_form(audio_path, whisper_pipe, chunk_length_s=15)
    rprint(f"[white]{text_long}[/white]")
    console.print()

    console.rule("[bold]Batch Transcription Example[/bold]")
    batch_texts = transcribe_batch([audio_path], whisper_pipe)
    for i, text in enumerate(batch_texts, 1):
        rprint(f"[cyan]File {i}:[/cyan] {text}")