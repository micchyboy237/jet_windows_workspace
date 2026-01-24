# transcribe_jp.py

import os
import io
import numpy as np
import numpy.typing as npt
import torch
import librosa
import threading
from dataclasses import asdict, dataclass
from typing import Iterator, Tuple, List, Union, Optional, TypedDict, Literal, Any
from rich.console import Console
from rich.logging import RichHandler
from rich.live import Live                    # new
from rich.text import Text                    # new
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from utils import split_sentences_ja
from warnings import warn

console = Console()
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float

    def _asdict(self):
        warn(
            "Word._asdict() method is deprecated, use dataclasses.asdict(Word) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]               # or List[dict] if you keep fake dict tokens
    avg_logprob: Optional[float] = None
    temperature: Optional[float] = None
    # compression_ratio: Optional[float] = None
    # no_speech_prob: Optional[float] = None
    # words: Optional[List[Word]] = None

    def _asdict(self):
        warn(
            "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


class TokenDetail(TypedDict, total=False):
    token: str
    logprob: float
    token_id: int
    # future: start_frame: int, duration_frames: int, start_time_s: float, end_time_s: float


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

# Short audio example
DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

# Long audio example
# DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

MODEL_ID = "reazon-research/japanese-wav2vec2-large-rs35kh"
TARGET_SR = 16_000
DEFAULT_MAX_CHUNK_SEC = 15.0       # Increased — better for Japanese sentence continuity
DEFAULT_CHUNK_OVERLAP_SEC = 3.0   # Much better boundary context (≈22% overlap)

# Quality categorization thresholds (empirical - Japanese wav2vec2 models)
QUALITY_THRESHOLDS_AVG_LOGPROB = [
    (-0.50, "very_high"),
    (-0.95, "high"),
    (-1.55, "medium"),
    (-2.50, "low"),
    (float("-inf"), "very_low"),
]

QUALITY_THRESHOLDS_SEQ_LOGPROB = [
    (-20.0, "very_high"),
    (-45.0, "high"),
    (-85.0, "medium"),
    (-160.0, "low"),
    (float("-inf"), "very_low"),
]


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
    token_details: Optional[List[TokenDetail]]

    # Quality category labels
    quality_avg_logprob: Optional[QualityCategory]
    quality_sequence_logprob: Optional[QualityCategory]


def chunk_audio(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    max_chunk_seconds: float,
    overlap_seconds: float,
) -> List[npt.NDArray[np.float32]]:
    """
    Split a 1D float32 audio array into overlapping chunks suitable for ASR.

    Returns slices/views of the original array (zero-copy where possible).
    """
    if len(audio) == 0:
        return []

    if max_chunk_seconds <= 0 or overlap_seconds < 0:
        raise ValueError("max_chunk_seconds must be > 0 and overlap_seconds >= 0")

    chunk_size = int(max_chunk_seconds * sample_rate)
    overlap = int(overlap_seconds * sample_rate)
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError(
            f"overlap ({overlap_seconds}s) must be strictly less than "
            f"max_chunk_seconds ({max_chunk_seconds}s) at sample rate {sample_rate}"
        )

    chunks: List[npt.NDArray[np.float32]] = []
    start = 0

    while start < len(audio):
        end = min(start + chunk_size, len(audio))
        chunks.append(audio[start:end])
        start += step

    return chunks


class JapaneseASR:
    """Flexible Japanese speech-to-text using wav2vec2-large-xlsr-53-japanese"""

    @staticmethod
    def _get_quality_label(
        value: float, thresholds: List[Tuple[float, QualityCategory]]
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
                MODEL_ID,
                local_files_only=True,
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                MODEL_ID,
                local_files_only=True,
            )
            self.model.to(self.device)
            self.model.eval()

        logger.info("[green]✓ Model & processor loaded successfully[/green]")

    def _transcribe_segment(
        self,
        segment: npt.NDArray[np.float32],
        return_confidence: bool = False,
        return_logprobs: bool = False,
        return_token_details: bool = False,
    ) -> Tuple[str, Optional[float], Optional[float], Optional[List[float]], Optional[List[TokenDetail]]]:
        """Core transcription logic for one contiguous segment (short or chunk)."""
        if len(segment) == 0:
            return "", None, None, None, None

        inputs = self.processor(
            segment, sampling_rate=TARGET_SR, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_outputs = self.model(inputs.input_values, attention_mask=inputs.attention_mask)
            logits = model_outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)

        text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        avg_logprob = None
        seq_logprob = None
        token_logprobs = None
        token_details: Optional[List[TokenDetail]] = None

        if return_confidence or return_logprobs:
            log_probs = torch.log_softmax(logits, dim=-1)
            best_token_logprobs = log_probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)[0]

            avg_logprob = best_token_logprobs.mean().item()
            seq_logprob = best_token_logprobs.sum().item()

            if return_logprobs:
                token_logprobs = best_token_logprobs.cpu().numpy().tolist()

                if return_token_details:
                    token_ids = pred_ids[0].cpu().tolist()
                    tokens = self.processor.tokenizer.convert_ids_to_tokens(token_ids)

                    blank_token = self.processor.tokenizer.word_delimiter_token or "|"
                    pad_token   = self.processor.tokenizer.pad_token or "<pad>"
                    unk_token   = self.processor.tokenizer.unk_token or "<unk>"

                    token_details = []
                    for tok_str, lp, tok_id in zip(tokens, best_token_logprobs, token_ids):
                        if tok_str in {blank_token, pad_token, unk_token, ""}:
                            continue
                        token_details.append({
                            "token": tok_str,
                            "logprob": round(lp.item(), 5),
                            "token_id": int(tok_id),
                        })

        return text, avg_logprob, seq_logprob, token_logprobs, token_details
        # Note: we return 5 values now (added token_details as fifth)

    def _stitch_texts_with_overlap(self, prev: str, curr: str) -> str:
        """Naive but effective stitching for Japanese (no word boundaries)."""
        if not prev:
            return curr
        if not curr:
            return prev

        # Limit search window to avoid bad matches and keep it fast
        max_match_len = 40
        prev_end = prev[-max_match_len:]
        curr_start = curr[:max_match_len]

        best_overlap = 0
        for ol in range(min(len(prev_end), len(curr_start)), 3, -1):  # min overlap 4 chars
            if prev_end[-ol:] == curr_start[:ol]:
                best_overlap = ol
                break

        if best_overlap >= 4:
            stitched = prev + curr[best_overlap:]
            logger.debug(f"Stitched with overlap {best_overlap} chars: ...{prev[-15:]} + {curr[:15+best_overlap]}...")
            return stitched
        else:
            logger.debug("No good overlap match → fallback concat with space")
            return prev + " " + curr

    def transcribe(
        self,
        audio: AudioInput,
        input_sample_rate: Optional[int] = None,
        return_confidence: bool = False,
        return_logprobs: bool = False,
        num_beams: int = 1,
        return_token_details: bool = False,
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
                    array, orig_sr = librosa.load(audio, sr=TARGET_SR, mono=True)
            except Exception as e:
                raise ValueError(f"Failed to load audio file {source_name}") from e

        elif isinstance(audio, bytes):
            source_name = "<bytes>"
            try:
                with self.console.status("[cyan]Loading raw PCM bytes...", spinner="arc"):
                    # ─── Quick & reliable way ───
                    array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

                    # Optional: very basic silence trimming (helps a bit)
                    energy = np.abs(array)
                    mask = energy > 0.015   # ~500 / 32768
                    if np.any(mask):
                        first = np.where(mask)[0][0]
                        last = np.where(mask)[0][-1] + 1
                        array = array[first:last]
                    else:
                        array = np.array([], dtype=np.float32)

                orig_sr = input_sample_rate
                if orig_sr is None:
                    raise ValueError("For raw PCM bytes you must provide input_sample_rate!")
            except Exception as e:
                raise ValueError("Failed to interpret bytes as raw 16-bit PCM") from e

        elif isinstance(audio, np.ndarray):
            source_name = "<numpy array>"
            array = audio.astype(np.float32)
            orig_sr = input_sample_rate
            if orig_sr is None:
                raise ValueError("When passing numpy array, you must provide input_sample_rate=...")

        elif isinstance(audio, torch.Tensor):
            source_name = "<torch.Tensor>"
            array = audio.cpu().numpy().astype(np.float32)
            orig_sr = input_sample_rate
            if orig_sr is None:
                raise ValueError("When passing torch.Tensor, you must provide input_sample_rate=...")

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

        if len(array) == 0:
            return {
                "text": "",
                "file": source_name,
                "duration_sec": 0.0,
                "chunks_info": "empty input (0 samples)",
            }

        # Resample only if needed (librosa.load already did it for file/bytes)
        if orig_sr != TARGET_SR:
            logger.info(f"Resampling from {orig_sr}Hz → {TARGET_SR}Hz")
            array = librosa.resample(array, orig_sr=orig_sr, target_sr=TARGET_SR)

        duration_sec = len(array) / TARGET_SR
        logger.info(f"Audio duration: [bold]{duration_sec:.1f} seconds[/bold]")

        result: TranscriptionResult = {
            "text": "",
            "file": source_name,
            "duration_sec": round(duration_sec, 1),
        }

        # ── Short audio: single pass ──────────────────────────────────────
        if duration_sec <= max_chunk_sec + 1.0:
            result["chunks_info"] = "single chunk"

            text, avg_logprob, seq_logprob, token_logprobs, token_details = self._transcribe_segment(
                array,
                return_confidence=return_confidence,
                return_logprobs=return_logprobs,
                return_token_details=return_token_details
            )

            result["text"] = text

            result["avg_logprob"] = round(avg_logprob, 5) if avg_logprob is not None else None
            result["sequence_logprob"] = round(seq_logprob, 5) if seq_logprob is not None else None

            if return_confidence or return_logprobs:
                if avg_logprob is not None and seq_logprob is not None:
                    result["avg_confidence"] = round(np.exp(avg_logprob), 4)

                if return_logprobs:
                    result["token_logprobs"] = token_logprobs
                    if return_token_details:
                        result["token_details"] = token_details

                if "avg_logprob" in result and result["avg_logprob"] is not None:
                    result["quality_avg_logprob"] = self._get_quality_label(
                        result["avg_logprob"], QUALITY_THRESHOLDS_AVG_LOGPROB
                    )
                if "sequence_logprob" in result and result["sequence_logprob"] is not None:
                    result["quality_sequence_logprob"] = self._get_quality_label(
                        result["sequence_logprob"], QUALITY_THRESHOLDS_SEQ_LOGPROB
                    )

            return result

        # ── Long audio → chunked processing ───────────────────────────────
        else:
            audio_chunks = chunk_audio(
                array,
                TARGET_SR,
                max_chunk_sec,
                overlap_sec,
            )

            chunks_texts: List[str] = []
            total_avg_logprobs: List[float] = []
            total_seq_logprobs: float = 0.0
            chunk_count = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("[magenta]Chunked transcription...", total=len(audio_chunks))

                for chunk in audio_chunks:
                    if num_beams > 1:
                        logger.warning("Beam search not supported in chunked mode → using greedy")

                    text, avg_lp, seq_lp, _, _ = self._transcribe_segment(
                        chunk,
                        return_confidence=return_confidence,
                        return_logprobs=return_logprobs,
                        return_token_details=False   # ← not collecting per-chunk details yet
                    )
                    chunks_texts.append(text.strip())

                    if avg_lp is not None:
                        total_avg_logprobs.append(avg_lp)
                    if seq_lp is not None:
                        total_seq_logprobs += seq_lp
                        chunk_count += 1

                    progress.advance(task)

            # ── Stitch chunks using overlap-aware concatenation ───────────
            if not chunks_texts:
                full_text = ""
            elif len(chunks_texts) == 1:
                full_text = chunks_texts[0]
            else:
                full_text = chunks_texts[0]
                for i in range(1, len(chunks_texts)):
                    full_text = self._stitch_texts_with_overlap(full_text, chunks_texts[i])

            full_text = full_text.strip()
            result["text"] = full_text
            result["chunks_info"] = f"{len(audio_chunks)} chunks (overlap {overlap_sec:.1f}s, with text stitching)"

            if total_avg_logprobs:
                avg_of_avgs = sum(total_avg_logprobs) / len(total_avg_logprobs)
                result["avg_logprob"] = round(avg_of_avgs, 5)
                result["sequence_logprob"] = round(total_seq_logprobs, 5) if chunk_count > 0 else None

                result["quality_avg_logprob"] = self._get_quality_label(
                    avg_of_avgs, QUALITY_THRESHOLDS_AVG_LOGPROB
                )
                if result["sequence_logprob"] is not None:
                    result["quality_sequence_logprob"] = self._get_quality_label(
                        result["sequence_logprob"], QUALITY_THRESHOLDS_SEQ_LOGPROB
                    )

            return result

    def transcribe_stream(
        self,
        audio: AudioInput,
        input_sample_rate: Optional[int] = None,
        max_chunk_seconds: Optional[float] = None,
        chunk_overlap_seconds: Optional[float] = None,
    ) -> Iterator[Tuple[str, float]]:
        """
        Generator version: yields (current_stitched_text_so_far, progress_0_to_1)
        after each chunk is processed.

        Last yield has progress == 1.0
        """
        max_chunk_sec = max_chunk_seconds if max_chunk_seconds is not None else self.max_chunk_seconds
        overlap_sec   = chunk_overlap_seconds if chunk_overlap_seconds is not None else self.chunk_overlap_seconds

        # ── Same audio loading logic as transcribe() ───────────────────────
        if isinstance(audio, (str, os.PathLike)):
            source_name = str(audio)
            array, orig_sr = librosa.load(audio, sr=TARGET_SR, mono=True)
        elif isinstance(audio, bytes):
            source_name = "<bytes>"
            array, orig_sr = librosa.load(io.BytesIO(audio), sr=TARGET_SR, mono=True)
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
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        if orig_sr != TARGET_SR:
            array = librosa.resample(array, orig_sr=orig_sr, target_sr=TARGET_SR)

        if array.ndim > 1:
            array = np.mean(array, axis=1)

        duration_sec = len(array) / TARGET_SR
        if duration_sec == 0:
            yield "", 1.0
            return

        if duration_sec <= max_chunk_sec + 1.0:
            text, _, _, _, _ = self._transcribe_segment(array, False, False, False)
            yield text.strip(), 1.0
            return

        # Long audio
        chunks = chunk_audio(array, TARGET_SR, max_chunk_sec, overlap_sec)

        current_text = ""
        processed_sec = 0.0

        for chunk in chunks:
            text, _, _, _, _ = self._transcribe_segment(chunk, False, False, False)
            text = text.strip()

            if not current_text:
                current_text = text
            else:
                current_text = self._stitch_texts_with_overlap(current_text, text)

            processed_sec += max_chunk_sec - overlap_sec
            progress = min(1.0, processed_sec / duration_sec)

            yield current_text, progress

        # Final clean yield
        yield current_text, 1.0

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


_default_asr_instance: JapaneseASR | None = None
_asr_lock = threading.Lock()

def get_japanese_asr() -> JapaneseASR:
    global _default_asr_instance

    if _default_asr_instance is not None:
        return _default_asr_instance

    with _asr_lock:
        # double check
        if _default_asr_instance is not None:
            return _default_asr_instance

        logger.info("Initializing Japanese ASR (once only)...")
        instance = JapaneseASR(
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_chunk_seconds=45.0,
            chunk_overlap_seconds=10.0,
        )
        logger.info("ASR model ready ✓")
        _default_asr_instance = instance
        return instance


def transcribe_with_japanese_asr(
    audio_bytes: bytes,
    sample_rate: int,
    language: str = "ja",               # currently ignored – model is ja only
    beam_size: int = 1,                 # currently not supported
    vad_filter: bool = False,           # ignored
    condition_on_previous_text: bool = False,  # ignored
    **kwargs
) -> tuple[Segment, dict, Optional[List[TokenDetail]]]:
    """
    Adapter to make JapaneseASR behave similarly to faster-whisper .transcribe()
    Returns dict that tries to be compatible with existing server code
    """
    asr = get_japanese_asr()

    # JapaneseASR already handles bytes very well
    result: TranscriptionResult = asr.transcribe(
        audio=audio_bytes,
        input_sample_rate=sample_rate,
        return_confidence=True,
        return_logprobs=True,           # we don't use per-token yet
        return_token_details=True,
        num_beams=beam_size,             # currently ignored
    )

    # ── Convert to fake "segments" format expected by server ─────────────
    full_text = result["text"]
    sentences = split_sentences_ja(full_text)  # assuming you import this
    token_ids = [td["token_id"] for td in result.get("token_details") or []]
    # logprobs = [td["logprob"] for td in result.get("token_details") or []]
    # tokens = [td["token"] for td in result.get("token_details") or []]

    segments = []
    current_time = 0.0
    duration = result["duration_sec"]

    if sentences:
        time_per_sentence = duration / len(sentences)
        for sent in sentences:
            segment_obj = Segment(
                id=len(segments),              # just incremental
                seek=0,                        # almost always 0 for live/short segments
                start=current_time,
                end=current_time + time_per_sentence,
                text=sent,
                tokens=token_ids,  # ← note: this is List[dict], not List[int]
                avg_logprob=result.get("avg_logprob"),
                temperature=0.0,               # fake value - most servers ignore it anyway
            )
            segments.append(segment_obj)
            current_time += time_per_sentence

    info = {
            "language": "ja",
            "language_probability": 0.99,  # dummy
            "duration": duration,
        }
    return segments, info, result.get("token_details")


# ────────────────────────────────────────
# Example usage
# ────────────────────────────────────────
def example(audio_path):
    asr = JapaneseASR()

    try:
        result = asr.transcribe(
            audio_path,
            return_confidence=True,
            return_logprobs=True,
            return_token_details=True,          # ← added to show token + logprob table
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
            color = {
                "very_high": "bright_green",
                "high": "green",
                "medium": "yellow",
                "low": "orange1",
                "very_low": "red"
            }.get(cat, "white")
            asr.console.print(
                f"[bold]Avg token log-prob:[/bold] {alp:.4f} → [[{color}]{cat}[/{color}]]"
            )

        if (slp := result.get("sequence_logprob")) is not None:
            cat = result.get("quality_sequence_logprob", "—")
            color = {
                "very_high": "bright_green",
                "high": "green",
                "medium": "yellow",
                "low": "orange1",
                "very_low": "red"
            }.get(cat, "white")
            asr.console.print(
                f"[bold]Sequence log-prob:[/bold] {slp:.3f} → [[{color}]{cat}[/{color}]]"
            )

        if result.get("token_logprobs"):
            asr.console.print(f"[dim]Token logprobs (first 15):[/dim] {result['token_logprobs'][:15]}")

        asr.console.print("")  # small spacing

        if result.get("token_details"):
            asr.console.rule("Token-Level Details (first 20 non-blank)", style="dim")
            from rich.table import Table
            table = Table(title="First 20 Predicted Tokens", show_header=True, expand=False)
            table.add_column("Token", style="cyan", no_wrap=True)
            table.add_column("Log-prob", style="magenta", justify="right")
            table.add_column("Token ID", style="dim blue", justify="right")
            for item in (result["token_details"] or [])[:20]:
                table.add_row(
                    item.get("token", "—"),
                    f"{item.get('logprob', 0.0):.5f}",
                    str(item.get("token_id", "—"))
                )
            asr.console.print(table)

    except Exception:
        logger.exception("Transcription failed")


def live_stream_example(audio_path: str):
    asr = JapaneseASR()

    console.rule(f"[bold cyan]Live Streaming Transcription – {Path(audio_path).name}[/bold cyan]")


    last_text = None
    for partial_text, progress in asr.transcribe_stream(audio_path):
        if partial_text != last_text or progress >= 1.0:
            ja_text = partial_text if partial_text else "[dim](processing first chunk...)[/dim]"
            ja_sents = split_sentences_ja(ja_text)
            # en_text = translate_text(ja_text)
            last_text = partial_text

            yield ja_sents, progress

    console.print("\n[green bold]Streaming complete.[/green bold]")


if __name__ == "__main__":
    from rich import print as rprint

    from translate_jp_en_llm import translate_text

    AUDIO_SHORT = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
    AUDIO_LONG  = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    rprint("\n[bold magenta]Demo: Long Audio – Streaming / Live mode[/bold magenta]")
    results_stream = live_stream_example(AUDIO_LONG)

    with Live(refresh_per_second=3, console=console) as live:
        for ja_sents, progress in results_stream:
            ja_sents_text = "\n".join(f"# {num}: {sent}" for num, sent in enumerate(ja_sents, start=1))
            en_sents = []
            for ja_sent in ja_sents:
                en_sent = translate_text(ja_sent)
                en_sents.append(ja_sent)
                
                live.update(
                    Text.from_markup(f"[cyan]{progress:>6.1%}[/cyan]\nJA: {ja_sents_text}\nEN: {en_sent}")
                )


    # rprint("[bold cyan]Demo: Short Audio (normal mode)[/bold cyan]")
    # example(AUDIO_SHORT)

    # rprint("\n[bold cyan]Demo: Long Audio (normal mode)[/bold cyan]")
    # example(AUDIO_LONG)
