import torch
from transformers import pipeline
from typing import List, Dict, Any

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich import print as rprint

# Global rich console for consistent output
console = Console()

# Model configuration (shared across all functions)
MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0"
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
        result = pipe(audio_path, generate_kwargs=GENERATE_KWARGS)
        progress.update(task, completed=100)
    
    console.print("[bold green]Transcription complete[/bold green]")
    return result["text"]


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
        result = pipe(
            audio_path,
            return_timestamps=True,
            generate_kwargs=GENERATE_KWARGS,
        )
        progress.update(task, completed=100)
    
    console.print("[bold green]Timestamped transcription complete[/bold green]")
    return result["chunks"]


def transcribe_long_form(audio_path: str, pipe: Any, chunk_length_s: int = 15) -> str:
    """
    Perform transcription suitable for longer audio files using chunking.
    
    Args:
        audio_path: Path to the WAV file.
        pipe: Initialized transformers pipeline.
        chunk_length_s: Length of each processing chunk in seconds (default 15).
    
    Returns:
        Full transcribed text.
    """
    console.rule(f"[bold magenta]Long-form Transcription (chunk_length_s={chunk_length_s})[/bold magenta]")
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
            chunk_length_s=chunk_length_s,
            generate_kwargs=GENERATE_KWARGS,
        )
        progress.update(task, completed=100)
    
    console.print("[bold green]Long-form transcription complete[/bold green]")
    return result["text"]


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
        results = pipe(audio_paths, generate_kwargs=GENERATE_KWARGS)
        progress.update(task, completed=100)
    
    console.print("[bold green]Batch transcription complete[/bold green]")
    return [r["text"] for r in results]


if __name__ == "__main__":
    # Single audio file to test all examples
    AUDIO_FILE = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\sound.wav"

    # Create pipeline once (model download happens here on first run)
    whisper_pipe = create_pipeline()

    console.rule("[bold]Demo: Kotoba-Whisper v2.0 on Local Audio[/bold]")
    text_basic = transcribe_basic(AUDIO_FILE, whisper_pipe)
    rprint(f"[white]{text_basic}[/white]")
    console.print()

    console.rule("[bold]Transcription with Timestamps[/bold]")
    chunks = transcribe_with_timestamps(AUDIO_FILE, whisper_pipe)
    for chunk in chunks:
        start, end = chunk["timestamp"]
        rprint(f"[yellow][{start:.2f}s → {end:.2f}s][/yellow] {chunk['text']}")
    console.print()

    console.rule("[bold]Long-form Transcription[/bold]")
    text_long = transcribe_long_form(AUDIO_FILE, whisper_pipe, chunk_length_s=15)
    rprint(f"[white]{text_long}[/white]")
    console.print()

    console.rule("[bold]Batch Transcription Example[/bold]")
    batch_texts = transcribe_batch([AUDIO_FILE], whisper_pipe)
    for i, text in enumerate(batch_texts, 1):
        rprint(f"[cyan]File {i}:[/cyan] {text}")