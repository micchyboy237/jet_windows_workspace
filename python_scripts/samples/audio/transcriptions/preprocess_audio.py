import os
import shutil
import librosa
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from utils import save_file  # kept for compatibility if used elsewhere

# Rich imports
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.panel import Panel
from rich.traceback import install

import logging

# Install rich traceback handler (prettier errors)
install()

# Setup console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
log = logging.getLogger("rich")

# Output directory setup
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

log.info(f"[bold green]Output directory ready:[/bold green] {OUTPUT_DIR}")


def preprocess_librosa(input_path: str, output_path: str) -> None:
    """Preprocess audio: resample to 16kHz mono, trim silence, normalize."""
    log.info(f"Loading audio from: [cyan]{input_path}[/cyan]")
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[bold blue]Preprocessing audio...", total=100)

        # Load and resample
        progress.update(task, advance=30, description="Loading & resampling to 16kHz mono...")
        y, sr = librosa.load(input_path, sr=16000, mono=True)

        # Trim silence
        progress.update(task, advance=30, description="Trimming leading/trailing silence...")
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # Normalize
        progress.update(task, advance=30, description="Normalizing amplitude...")
        y_norm = librosa.util.normalize(y_trimmed)

        # Save
        progress.update(task, advance=10, description="Saving cleaned audio...")
        sf.write(output_path, y_norm, 16000, subtype="PCM_16")
    
    log.info(f"[bold green]Preprocessed audio saved:[/bold green] {output_path}")


def transcribe_with_faster_whisper(model_path: str, clean_path: str, device: str = "cuda") -> str:
    """Transcribe preprocessed audio using faster-whisper."""
    if not os.path.exists(model_path):
        log.error(f"[bold red]Model path does not exist:[/bold red] {model_path}")
        raise FileNotFoundError(model_path)

    log.info(f"Loading Whisper model: [bold magenta]{os.path.basename(model_path)}[/bold magenta] on [bold yellow]{device}[/bold yellow]")

    model = WhisperModel(model_path, device=device, compute_type="int8_float16" if "int8" in model_path else "float16")
    
    log.info(f"Starting transcription of: [cyan]{clean_path}[/cyan]")
    
    # Use rich progress for transcription (faster-whisper yields segments)
    segments_text = []
    with Progress(
        TextColumn("[bold green]Transcribing...[/bold green] {task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing segments", total=None)  # unknown total

        segments, info = model.transcribe(clean_path, beam_size=5, language="en", vad_filter=True)
        
        for i, segment in enumerate(segments):
            segments_text.append(segment.text.strip())
            progress.update(task, description=f"Segment {i+1}: {segment.text[:60]}...")

    full_text = " ".join(segments_text).strip()
    
    log.info(f"[bold green]Transcription complete![/bold green] Detected language: [yellow]{info.language}[/yellow] "
             f"Probability: [yellow]{info.language_probability:.2f}[/yellow] Duration: [yellow]{info.duration:.2f}s[/yellow]")
    
    return full_text


if __name__ == "__main__":
    # === Configuration ===
    model_path = r"C:\Users\druiv\.cache\hf_ctranslate2_models\faster-whisper-large-v3-int8_float16"
    input_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\1.wav"
    preprocessed_path = os.path.join(OUTPUT_DIR, "cleaned.wav")

    # Header
    console.rule("[bold blue]Faster Whisper Transcription with Rich Logging[/bold blue]")
    rprint(Panel.fit(f"[bold]Input:[/bold] {input_path}\n[bold]Model:[/bold] large-v3-int8_float16", title="Config"))

    try:
        # Step 1: Preprocess
        preprocess_librosa(input_path, preprocessed_path)

        # Step 2: Transcribe
        text = transcribe_with_faster_whisper(model_path, preprocessed_path, device="cuda")

        # Final result
        console.rule("[bold green]Final Transcription[/bold green]")
        rprint(Panel(text, title="Output", border_style="bright_blue", padding=1))

        # Optional: save transcription
        txt_path = os.path.join(OUTPUT_DIR, "transcription.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        log.info(f"Transcription saved to: [cyan]{txt_path}[/cyan]")

    except Exception as e:
        log.exception(f"[bold red]An error occurred:[/bold red] {e}")