import os
import io
from typing import List, Union, Optional, Dict, TypedDict, Literal, Any
import numpy as np
import numpy.typing as npt
import torch
import librosa
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from audio_utils import load_audio_bytes

# Audio input type: file-like, np/tensor, bytes, etc
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    torch.Tensor,
]

# Setup rich-based logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)]
)
logger = logging.getLogger("JapaneseASR")


QualityCategory = Literal["very_low", "low", "medium", "high", "very_high"]


class TranscriptionResult(TypedDict, total=False):
    text: str
    file: str
    chunks_info: str
    duration_sec: float

    # Confidence & log-prob related
    avg_confidence: Optional[float]
    avg_logprob: Optional[float]
    sequence_logprob: Optional[float]
    token_logprobs: Optional[List[float]]
    best_beam_score: Optional[float]

    # Quality category labels
    quality_avg_logprob: Optional[QualityCategory]
    quality_sequence_logprob: Optional[QualityCategory]


class JapaneseASR:
    """Flexible Japanese speech-to-text using wav2vec2-large-xlsr-53-japanese"""

    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
    TARGET_SR = 16_000
    DEFAULT_MAX_CHUNK_SEC = 30.0
    DEFAULT_CHUNK_OVERLAP_SEC = 2.0

    # Quality categorization thresholds (empirical - Japanese wav2vec2 models)
    # avg_logprob: roughly per-token average log probability
    QUALITY_THRESHOLDS_AVG_LOGPROB = [
        (-0.50, "very_high"),
        (-0.95, "high"),
        (-1.55, "medium"),
        (-2.50, "low"),
        (float("-inf"), "very_low"),
    ]

    # sequence_logprob: total sum - much more length dependent
    QUALITY_THRESHOLDS_SEQ_LOGPROB = [
        (-20.0, "very_high"),
        (-45.0, "high"),
        (-85.0, "medium"),
        (-160.0, "low"),
        (float("-inf"), "very_low"),
    ]

    @staticmethod
    def _get_quality_label(
        value: float, thresholds: List[tuple[float, QualityCategory]]
    ) -> QualityCategory:
        for threshold, label in thresholds:
            if value >= threshold:
                return label
        return "very_low"  # fallback

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

    def transcribe(
        self,
        audio: AudioInput,
        input_sample_rate: Optional[int] = None,   # only needed when passing raw array/tensor
        return_confidence: bool = False,
        return_logprobs: bool = False,
        num_beams: int = 1,
        max_chunk_seconds: Optional[float] = None,
        chunk_overlap_seconds: Optional[float] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Flexible Japanese speech-to-text.

        Accepts: path (str/Path), bytes, numpy array, torch.Tensor
        """
        # ── Resolve audio array & sample rate ─────────────────────────────
        if isinstance(audio, (str, os.PathLike)):
            source_name = str(audio)
            try:
                with self.console.status(f"[cyan]Loading {source_name}...", spinner="arc"):
                    array, orig_sr = librosa.load(audio, sr=self.TARGET_SR, mono=True)
            except Exception as e:
                raise ValueError(f"Failed to load audio file {source_name}") from e

        elif isinstance(audio, bytes):
            source_name = "<bytes>"
            # ── Most common live-streaming case: raw PCM ─────────────────────
            if input_sample_rate is None:
                raise ValueError(
                    "When passing raw bytes from live capture, you MUST provide "
                    "input_sample_rate= (usually 16000, 44100, 48000...)"
                )

            try:
                array, orig_sr = load_audio_bytes(
                    audio,
                    expected_sample_rate=input_sample_rate,
                    channels=1,               # ← change if you capture stereo
                    dtype=np.float32          # ← pyaudio usually float32 or int16
                )
            except ValueError as e:
                raise ValueError("Failed to interpret bytes as raw PCM audio") from e

        elif isinstance(audio, np.ndarray):
            source_name = "<numpy array>"
            array = audio.astype(np.float32)
            orig_sr = input_sample_rate
            if orig_sr is None:
                raise ValueError("input_sample_rate required for numpy array")

        elif isinstance(audio, torch.Tensor):
            source_name = "<torch.Tensor>"
            array = audio.cpu().numpy().astype(np.float32)
            orig_sr = input_sample_rate
            if orig_sr is None:
                raise ValueError("input_sample_rate required for torch.Tensor")

        else:
            raise TypeError(
                f"Unsupported audio type: {type(audio).__name__}\n"
                f"Expected one of: str, Path, bytes, numpy.ndarray, torch.Tensor"
            )

        logger.info(f"Processing: [blue]{source_name}[/blue]")

        max_chunk_sec = max_chunk_seconds if max_chunk_seconds is not None else self.max_chunk_seconds
        overlap_sec = chunk_overlap_seconds if chunk_overlap_seconds is not None else self.chunk_overlap_seconds

        # Make sure we have 1D array
        if array.ndim > 1:
            array = np.mean(array, axis=1)  # very naive downmix

        # Resample only if needed (librosa.load already did it for file/bytes)
        if orig_sr != self.TARGET_SR:
            logger.info(f"Resampling from {orig_sr}Hz → {self.TARGET_SR}Hz")
            array = librosa.resample(array, orig_sr=orig_sr, target_sr=self.TARGET_SR)

        duration_sec = len(array) / self.TARGET_SR
        logger.info(f"Audio duration: [bold]{duration_sec:.1f} seconds[/bold]")

        result: TranscriptionResult = {
            "text": "",
            "file": source_name,
            "duration_sec": round(duration_sec, 1),
        }

        # ── Short audio ─ single inference ────────────────────────────────
        if duration_sec <= max_chunk_sec + 1.0:
            result["chunks_info"] = "single chunk"

            inputs = self.processor(
                array, sampling_rate=self.TARGET_SR, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad(), self.console.status("[magenta]Running inference...", spinner="bouncingBall"):
                model_outputs = self.model(inputs.input_values, attention_mask=inputs.attention_mask)
                logits = model_outputs.logits
                pred_ids = torch.argmax(logits, dim=-1)

            transcription = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            result["text"] = transcription

            if return_confidence or return_logprobs:
                log_probs = torch.log_softmax(logits, dim=-1)
                best_token_logprobs = log_probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)

                avg_logprob = best_token_logprobs.mean().item()
                seq_logprob = best_token_logprobs.sum().item()

                result["avg_logprob"] = round(avg_logprob, 5)
                result["sequence_logprob"] = round(seq_logprob, 5)
                result["avg_confidence"] = round(best_token_logprobs.exp().mean().item(), 4)

                if return_logprobs:
                    result["token_logprobs"] = [float(x) for x in best_token_logprobs[0].cpu().numpy()]

                # ── Quality labels ──────────────────────────────────────
                if "avg_logprob" in result:
                    result["quality_avg_logprob"] = self._get_quality_label(
                        result["avg_logprob"], self.QUALITY_THRESHOLDS_AVG_LOGPROB
                    )
                if "sequence_logprob" in result:
                    result["quality_sequence_logprob"] = self._get_quality_label(
                        result["sequence_logprob"], self.QUALITY_THRESHOLDS_SEQ_LOGPROB
                    )

            return result

        # ── Long audio → chunked processing ───────────────────────────────
        else:
            chunk_size_samples = int(max_chunk_sec * self.TARGET_SR)
            overlap_samples = int(overlap_sec * self.TARGET_SR)
            step_samples = chunk_size_samples - overlap_samples

            chunks: List[str] = []
            total_avg_logprobs: List[float] = []

            start = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                total_steps = (len(array) + step_samples - 1) // step_samples
                task = progress.add_task("[magenta]Chunked transcription...", total=total_steps)

                while start < len(array):
                    end = min(start + chunk_size_samples, len(array))
                    chunk_array = array[start:end]

                    inputs = self.processor(
                        chunk_array, sampling_rate=self.TARGET_SR, return_tensors="pt", padding=True
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs.input_values,
                            attention_mask=inputs.attention_mask,
                            num_beams=num_beams,
                            return_dict_in_generate=True,
                            output_scores=True,
                        )
                        pred_ids = outputs.sequences
                        text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                        chunks.append(text)

                        # Greedy path logprobs only (most common case)
                        if (return_confidence or return_logprobs) and num_beams == 1:
                            out = self.model(inputs.input_values, attention_mask=inputs.attention_mask)
                            logits = out.logits
                            log_probs = torch.log_softmax(logits, dim=-1)
                            best_token_logprobs = log_probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
                            total_avg_logprobs.append(best_token_logprobs.mean().item())

                    progress.advance(task)
                    start += step_samples

            full_text = " ".join(chunks).strip()
            result["text"] = full_text
            result["chunks_info"] = f"{len(chunks)} chunks (overlap {overlap_sec:.1f}s)"

            if total_avg_logprobs:
                avg_of_avgs = sum(total_avg_logprobs) / len(total_avg_logprobs)
                result["avg_logprob"] = round(avg_of_avgs, 5)
                result["quality_avg_logprob"] = self._get_quality_label(
                    avg_of_avgs, self.QUALITY_THRESHOLDS_AVG_LOGPROB
                )

            return result

    def batch_transcribe(
        self,
        audio_paths: List[Union[str, os.PathLike]],
        show_progress: bool = True,
        **kwargs,
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files with progress bar"""
        results = []
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("[cyan]Transcribing files...", total=len(audio_paths))
                for path in audio_paths:
                    try:
                        result = self.transcribe(path, **kwargs)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed {path}: {e}")
                        results.append({"file": str(path), "text": "", "error": str(e)})
                    progress.advance(task)
        else:
            for path in audio_paths:
                results.append(self.transcribe(path, **kwargs))
        return results


# ────────────────────────────────────────
# Example usage
# ────────────────────────────────────────
if __name__ == "__main__":
    asr = JapaneseASR()

    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

    try:
        result = asr.transcribe(
            audio_path,
            return_confidence=True,
            return_logprobs=True,
            num_beams=1,
        )

        asr.console.rule("Transcription Result")
        asr.console.print(f"[bold]File:[/bold] {result['file']}")
        asr.console.print(f"[bold]Duration:[/bold] {result['duration_sec']} s")
        asr.console.print(f"[bold]Chunks:[/bold] {result['chunks_info']}")
        asr.console.print(f"[bold]Text:[/bold]\n{result['text']}")

        if (conf := result.get("avg_confidence")) is not None:
            color = "green" if conf >= 0.65 else "yellow" if conf >= 0.40 else "red"
            asr.console.print(f"\n[bold]Average confidence:[/bold] [{color}]{conf:.1%}[/{color}]")

        if (alp := result.get("avg_logprob")) is not None:
            cat = result.get("quality_avg_logprob", "—")
            color = {"very_high": "bright_green", "high": "green", "medium": "yellow",
                     "low": "orange1", "very_low": "red"}.get(cat, "white")
            asr.console.print(
                f"[bold]Avg token log-prob:[/bold] {alp:.4f} → [[{color}]{cat}[/{color}]]"
            )

        if (slp := result.get("sequence_logprob")) is not None:
            cat = result.get("quality_sequence_logprob", "—")
            color = {"very_high": "bright_green", "high": "green", "medium": "yellow",
                     "low": "orange1", "very_low": "red"}.get(cat, "white")
            asr.console.print(
                f"[bold]Sequence log-prob:[/bold] {slp:.3f} → [[{color}]{cat}[/{color}]]"
            )

        if result.get("token_logprobs"):
            asr.console.print(f"[dim]Token logprobs (first 15):[/dim] {result['token_logprobs'][:15]}")

    except Exception as e:
        logger.exception("Transcription failed")