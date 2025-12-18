from typing import List, TypedDict
from pathlib import Path
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

class SpeechSpeakerSegment(TypedDict):
    idx: int
    start: float
    end: float
    speaker: str
    duration: float
    prob: float  # Placeholder (1.0); pyannote does not provide per-segment probability

@torch.no_grad()
def extract_speech_speakers(
    audio: str | Path | np.ndarray | torch.Tensor,
    threshold: float = 0.5,  # Ignored (for API compatibility with VAD); pyannote uses internal thresholds
    sampling_rate: int = 16000,
    time_resolution: int = 2,
) -> List[SpeechSpeakerSegment]:
    """
    Extract speaker diarization segments using pyannote-audio.

    Args:
        audio: Path to audio file, or NumPy/Torch array (mono, float32 normalized).
        threshold: Unused (pyannote has no direct equivalent).
        sampling_rate: Audio sampling rate (default 16000 Hz).
        time_resolution: Decimal places for timestamps (default 2).

    Returns:
        List of speaker segments with timestamps in seconds.
    """
    # Load model with status
    with console.status("[bold green]Loading pyannote speaker diarization model...[/bold green]"):
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
            )
            console.print("✅ Pyannote model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}. Ensure model conditions accepted at https://hf.co/pyannote/speaker-diarization-community-1")

    # Auto GPU if available
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        console.print("🚀 Using GPU for inference")

    # Audio input handling (mirrors Silero logic for reusability)
    # --- REPLACED: file loading branch with forced mono handlng ---
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)

        # === FIX: Force mono by averaging channels if stereo/multi-channel ===
        if waveform.shape[0] > 1:
            console.print(f"[yellow]⚠️  Audio has {waveform.shape[0]} channels → downmixing to mono[/yellow]")
            waveform = waveform.mean(dim=0, keepdim=True)  # shape becomes (1, N)

        if sr != sampling_rate:
            console.print(f"[dim]Resampling from {sr} Hz → {sampling_rate} Hz[/dim]")
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)

        audio = waveform.squeeze(0).numpy()  # now guaranteed 1D

    # --- REPLACED: np.ndarray handling with forced mono without torch conversion ---
    elif isinstance(audio, np.ndarray):
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Force mono if needed
        if audio.ndim > 1:
            if audio.shape[0] > 1:  # (channels, samples)
                console.print(f"[yellow]⚠️  NumPy input has {audio.shape[0]} channels → downmixing to mono[/yellow]")
                audio = audio.mean(axis=0)
            else:  # (1, samples) or higher dim singleton
                audio = audio.squeeze()

    # --- REPLACED: torch.Tensor handling with forced mono ---
    elif isinstance(audio, torch.Tensor):
        if audio.dtype == torch.int16:
            audio = audio.float() / 32768.0
        elif audio.dtype != torch.float32:
            audio = audio.float()

        # Force mono
        if audio.ndim > 1:
            if audio.shape[0] > 1:
                console.print(f"[yellow]⚠️  Torch input has {audio.shape[0]} channels → downmixing to mono[/yellow]")
                audio = audio.mean(dim=0)
            else:
                audio = audio.squeeze(dim=0)

        audio = audio.numpy()

    else:
        raise TypeError("audio must be torch.Tensor, np.ndarray, str or Path")

    # --- Final mono check still applies (safe now) ---
    if audio.ndim != 1:
        raise ValueError("Audio must be mono (1D array)")

    # === NEW: Convert to torch and prepare pyannote-compatible input ===
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio)  # (time,) tensor

    # Ensure float32 (pyannote requirement)
    if audio.dtype != torch.float32:
        audio = audio.float()

    # Unsqueeze channel dimension: (time,) → (1, time)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    console.print("✅ Audio prepared as torch tensor (mono, 16kHz)")

    # pyannote expects dict when waveform is provided
    pyannote_input = {"waveform": audio, "sample_rate": sampling_rate}

    # Warn on unused threshold
    if threshold != 0.5:
        console.print(f"[yellow]⚠️  Threshold {threshold} ignored; pyannote uses internal VAD thresholds.[/yellow]")

    # Run diarization with progress
    with Progress(
        SpinnerColumn(),
        "[bold blue]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running speaker diarization...", total=1)
        with ProgressHook() as hook:  # Built-in progress for local runs
            output = pipeline(pyannote_input, hook=hook)
        progress.update(task, completed=1)

    # Process output into enhanced segments
    enhanced: List[SpeechSpeakerSegment] = []
    # CORRECT: use the Annotation stored inside the DiarizeOutput
    for idx, (turn, _, speaker) in enumerate(
        output.speaker_diarization.itertracks(yield_label=True)
    ):
        duration_sec = round(turn.end - turn.start, 3)
        start_rounded = round(turn.start, time_resolution)
        end_rounded = round(turn.end, time_resolution)
        enhanced.append(
            SpeechSpeakerSegment(
                idx=idx,
                start=start_rounded,
                end=end_rounded,
                speaker=speaker,
                duration=duration_sec,
                prob=1.0,
            )
        )

    return enhanced

if __name__ == "__main__":
    # Example usage (replace with your HF token)
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")
    segments = extract_speech_speakers(
        audio_file,
        threshold=0.5,
        time_resolution=2,
    )
    console.print(f"\n[bold green]Speaker segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"speaker=[bold magenta]{seg['speaker']}[/bold magenta] "
            f"duration=[bold cyan]{seg['duration']}s[/bold cyan] "
            f"prob=[bold green]{seg['prob']:.3f}[/bold green]"
        )