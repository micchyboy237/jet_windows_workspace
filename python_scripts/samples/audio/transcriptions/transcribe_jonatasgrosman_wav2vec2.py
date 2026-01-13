from pathlib import Path
from typing import List, Union, Optional, Dict

import torch
import librosa
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import logging

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Setup rich-based logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)]
)
logger = logging.getLogger("JapaneseASR")


class JapaneseASR:
    """Flexible Japanese speech-to-text using wav2vec2-large-xlsr-53-japanese"""

    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
    TARGET_SR = 16_000

    DEFAULT_MAX_CHUNK_SEC = 30.0
    DEFAULT_CHUNK_OVERLAP_SEC = 2.0

    def _compute_avg_confidence(self, logits: torch.Tensor) -> float:
        """
        Very simple token-level confidence: average of max-probability per frame
        Returns value roughly between 0.0 ~ 1.0
        """
        probs = torch.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)  # [batch, seq_len]
        avg_conf = max_probs.mean().item()
        return round(avg_conf, 4)

    def __init__(
        self,
        device: Optional[str] = None,
        max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SEC,
        chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SEC,
    ):
        self.console = Console()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_chunk_seconds = max_chunk_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds

        logger.info(f"Loading model on device: [bold cyan]{self.device}[/bold cyan]")

        with self.console.status("[bold green]Loading processor and model...", spinner="dots"):
            self.processor = Wav2Vec2Processor.from_pretrained(
                self.MODEL_ID,
                local_files_only=True,
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.MODEL_ID,
                local_files_only=True,
            )
            self.model.to(self.device)
            self.model.eval()

        logger.info("[green]✓ Model & processor loaded successfully[/green]")

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        return_confidence: bool = False,
        max_chunk_seconds: Optional[float] = None,
        chunk_overlap_seconds: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Transcribe a single Japanese audio file.
        Automatically uses chunking for long audio.
        """
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info(f"Processing: [blue]{path.name}[/blue]")

        # Override defaults if provided
        max_chunk_sec = max_chunk_seconds if max_chunk_seconds is not None else self.max_chunk_seconds
        overlap_sec = chunk_overlap_seconds if chunk_overlap_seconds is not None else self.chunk_overlap_seconds

        # ── Load full audio once ───────────────────────────────────────
        with self.console.status(f"[cyan]Loading & resampling {path.name}...", spinner="arc"):
            speech_array, orig_sr = librosa.load(path, sr=self.TARGET_SR, mono=True)

        duration_sec = len(speech_array) / self.TARGET_SR
        logger.info(f"Audio duration: [bold]{duration_sec:.1f} seconds[/bold]")

        if duration_sec <= max_chunk_sec + 1.0:
            # Short audio → single pass
            inputs = self.processor(
                speech_array,
                sampling_rate=self.TARGET_SR,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad(), self.console.status("[magenta]Running inference...", spinner="bouncingBall"):
                outputs = self.model(
                    inputs.input_values,
                    attention_mask=inputs.attention_mask
                )
                logits = outputs.logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0].strip()

            confidence = None
            if return_confidence:
                confidence = self._compute_avg_confidence(logits)

            full_text = transcription
            chunks_info = "single chunk"

            result: Dict[str, any] = {
                "text": full_text,
                "file": str(path),
                "chunks_info": chunks_info,
                "duration_sec": round(duration_sec, 1),
            }

            if return_confidence:
                result["avg_confidence"] = confidence

            return result

        else:
            # Long audio → chunk with overlap
            chunk_size_samples = int(max_chunk_sec * self.TARGET_SR)
            overlap_samples = int(overlap_sec * self.TARGET_SR)
            step_samples = chunk_size_samples - overlap_samples

            chunks: List[str] = []

            start = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                total_steps = (len(speech_array) + step_samples - 1) // step_samples
                task = progress.add_task("[magenta]Chunked transcription...", total=total_steps)

                while start < len(speech_array):
                    end = min(start + chunk_size_samples, len(speech_array))
                    chunk_array = speech_array[start:end]

                    inputs = self.processor(
                        chunk_array,
                        sampling_rate=self.TARGET_SR,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)

                    with torch.no_grad():
                        logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits

                    pred_ids = torch.argmax(logits, dim=-1)
                    text = self.processor.batch_decode(pred_ids)[0].strip()

                    chunks.append(text)

                    progress.advance(task)
                    start += step_samples

            # Naive stitching: join with space (can be improved later)
            full_text = " ".join(chunks).strip()
            chunks_info = f"{len(chunks)} chunks (overlap {overlap_sec:.1f}s)"

            result: Dict[str, any] = {
                "text": full_text,
                "file": str(path),
                "chunks_info": chunks_info,
                "duration_sec": round(duration_sec, 1),
            }

            if return_confidence:
                result["avg_confidence"] = None

            return result

    def transcribe_files(
        self,
        audio_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[Dict[str, any]]:
        """Transcribe multiple audio files with progress bar"""
        results = []

        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=self.console
            )
            task = progress.add_task("[cyan]Transcribing files...", total=len(audio_paths))

            with progress:
                for path in audio_paths:
                    try:
                        result = self.transcribe_file(path)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed {path}: {e}")
                        results.append({"file": str(path), "text": "", "error": str(e)})
                    progress.advance(task)
        else:
            for path in audio_paths:
                results.append(self.transcribe_file(path))

        return results


# ────────────────────────────────────────
#          Example usage
# ────────────────────────────────────────

if __name__ == "__main__":
    asr = JapaneseASR()

    # Single file example
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
    try:
        result = asr.transcribe_file(
            audio_path,
            return_confidence=True,
            # You can override chunk size here if desired
            # max_chunk_seconds=20.0,
            # chunk_overlap_seconds=3.0,
        )

        asr.console.rule("Transcription Result")
        asr.console.print(f"[bold]File:[/bold] {result['file']}")
        asr.console.print(f"[bold]Duration:[/bold] {result['duration_sec']} s")
        asr.console.print(f"[bold]Chunks:[/bold] {result['chunks_info']}")
        asr.console.print(f"[bold]Text:[/bold]\n{result['text']}")

        # Show confidence when available (currently only single chunk mode)
        if result.get("avg_confidence") is not None:
            conf = result["avg_confidence"]
            color = "green" if conf >= 0.65 else "yellow" if conf >= 0.40 else "red"
            asr.console.print(f"\n[bold]Average confidence:[/bold] [{color}]{conf:.1%}[/{color}]")

    except Exception as e:
        logger.exception("Transcription failed")