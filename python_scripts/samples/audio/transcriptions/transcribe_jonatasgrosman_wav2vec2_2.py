# transcribe_jonatasgrosman_wav2vec2.py

from typing import Dict, Tuple, List, Union, Optional, TypedDict, Literal
import os
import io
import numpy as np
import numpy.typing as npt
import torch
import librosa
import logging
import sys
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from translators.translate_jp_en_opus import translate_japanese_to_english

# Audio input type: file-like, np/tensor, bytes, etc
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    torch.Tensor,
]

from typing import Iterator, Tuple

# Setup basic logging (replace rich logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("JapaneseASR")


QualityCategory = Literal["very_low", "low", "medium", "high", "very_high"]

# Short audio example
DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

# Long audio example
# DEFAULT_AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
TARGET_SR = 16_000
DEFAULT_MAX_CHUNK_SEC = 10.0       # Increased — better for Japanese sentence continuity
DEFAULT_CHUNK_OVERLAP_SEC = 2.0   # Much better boundary context (≈22% overlap)

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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_chunk_seconds = max_chunk_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds

        logger.info(f"Loading model on device: {self.device}")

        print("Loading processor and model...")
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

        logger.info("✓ Model & processor loaded successfully")

    def _transcribe_segment(
        self,
        segment: npt.NDArray[np.float32],
        return_confidence: bool,
        return_logprobs: bool,
    ) -> Tuple[str, Optional[float], Optional[float], Optional[List[float]]]:
        """Core transcription logic for one contiguous segment (short or chunk)."""
        if len(segment) == 0:
            return "", None, None, None

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

        if return_confidence or return_logprobs:
            log_probs = torch.log_softmax(logits, dim=-1)
            best_token_logprobs = log_probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)[0]

            avg_logprob = best_token_logprobs.mean().item()
            seq_logprob = best_token_logprobs.sum().item()

            if return_logprobs:
                token_logprobs = best_token_logprobs.cpu().numpy().tolist()

        return text, avg_logprob, seq_logprob, token_logprobs

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
        num_beams: int = 4,
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
                print(f"Loading {source_name}...")
                array, orig_sr = librosa.load(audio, sr=TARGET_SR, mono=True)
            except Exception as e:
                raise ValueError(f"Failed to load audio file {source_name}") from e

        elif isinstance(audio, bytes):
            source_name = "<bytes>"
            try:
                print("Loading audio from bytes...")
                array, orig_sr = librosa.load(io.BytesIO(audio), sr=TARGET_SR, mono=True)
            except Exception as e:
                raise ValueError("Failed to decode audio from bytes") from e

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

        logger.info(f"Processing: {source_name}")

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
        logger.info(f"Audio duration: {duration_sec:.1f} seconds")

        result: TranscriptionResult = {
            "text": "",
            "file": source_name,
            "duration_sec": round(duration_sec, 1),
        }

        # ── Short audio: single pass ──────────────────────────────────────
        if duration_sec <= max_chunk_sec + 1.0:
            result["chunks_info"] = "single chunk"

            text, avg_logprob, seq_logprob, token_logprobs = self._transcribe_segment(
                array, return_confidence, return_logprobs
            )
            result["text"] = text

            result["avg_logprob"] = round(avg_logprob, 5) if avg_logprob is not None else None
            result["sequence_logprob"] = round(seq_logprob, 5) if seq_logprob is not None else None

            if return_confidence or return_logprobs:
                if avg_logprob is not None and seq_logprob is not None:
                    result["avg_confidence"] = round(np.exp(avg_logprob), 4)

                if return_logprobs:
                    result["token_logprobs"] = token_logprobs

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

            print(f"Processing {len(audio_chunks)} chunks...")
            chunks_texts: List[str] = []
            total_avg_logprobs: List[float] = []
            total_seq_logprobs: float = 0.0
            chunk_count = 0

            for i, chunk in enumerate(audio_chunks, 1):
                if num_beams > 1:
                    logger.warning("Beam search not supported in chunked mode → using greedy")

                text, avg_lp, seq_lp, _ = self._transcribe_segment(
                    chunk, return_confidence, return_logprobs
                )
                chunks_texts.append(text.strip())

                if avg_lp is not None:
                    total_avg_logprobs.append(avg_lp)
                if seq_lp is not None:
                    total_seq_logprobs += seq_lp
                    chunk_count += 1
                print(f"Chunk {i}/{len(audio_chunks)} done", end="\r")
            print("\nAll chunks processed.")

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
            text, _, _, _ = self._transcribe_segment(array, False, False)
            yield text.strip(), 1.0
            return

        # Long audio
        chunks = chunk_audio(array, TARGET_SR, max_chunk_sec, overlap_sec)

        current_text = ""
        processed_sec = 0.0

        for chunk in chunks:
            text, _, _, _ = self._transcribe_segment(chunk, False, False)
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
            print(f"Transcribing {len(audio_paths)} files...")
            for idx, path in enumerate(audio_paths, 1):
                try:
                    result = self.transcribe(path, **kwargs)
                    results.append(result)
                    print(f"File {idx}/{len(audio_paths)} done: {result.get('file',path)}", end="\r")
                except Exception as e:
                    logger.error(f"Failed {path}: {e}")
                    results.append({"file": str(path), "text": "", "error": str(e)})
            print("\nAll batch files processed.")
        else:
            for path in audio_paths:
                results.append(self.transcribe(path, **kwargs))
        return results


# ────────────────────────────────────────
# Example usage
# ────────────────────────────────────────
def example(audio_path):
    asr = JapaneseASR()
    num_beams = 4

    try:
        result = asr.transcribe(
            audio_path,
            return_confidence=True,
            return_logprobs=True,
            num_beams=num_beams,
        )

        print("=" * 80)
        print("Transcription Result")
        print("=" * 80)
        print(f"File:          {result['file']}")
        print(f"Duration:      {result['duration_sec']} s")
        print(f"Chunks:        {result['chunks_info']}")
        print("Text:")
        print(result['text'])

        if (conf := result.get("avg_confidence")) is not None:
            color = "green" if conf >= 0.65 else "yellow" if conf >= 0.40 else "red"
            print(f"\nAverage confidence: [{color}]{conf:.1%}[/{color}]")

        if (alp := result.get("avg_logprob")) is not None:
            cat = result.get("quality_avg_logprob", "—")
            color = {"very_high": "bright_green", "high": "green", "medium": "yellow",
                     "low": "orange1", "very_low": "red"}.get(cat, "white")
            print(f"Avg token log-prob: {alp:.4f} → [{color}]{cat}[/{color}]")

        if (slp := result.get("sequence_logprob")) is not None:
            cat = result.get("quality_sequence_logprob", "—")
            color = {"very_high": "bright_green", "high": "green", "medium": "yellow",
                     "low": "orange1", "very_low": "red"}.get(cat, "white")
            print(f"Sequence log-prob:  {slp:.3f} → [{color}]{cat}[/{color}]")

        if result.get("token_logprobs"):
            print(f"Token logprobs (first 15): {result['token_logprobs'][:15]}")

    except Exception:
        logger.exception("Transcription failed")


def live_stream_example(audio_path: str):
    from pathlib import Path

    asr = JapaneseASR()

    print("=" * 80)
    print(f"Live Streaming Transcription + Translation – {Path(audio_path).name}")
    print("=" * 80)
    print()  # empty line

    # Accumulators
    full_japanese: str = ""
    full_english: str = ""
    last_translated_jp_len: int = 0

    print("Starting live transcription and translation...")
    print("-" * 80)

    for partial_jp, progress in asr.transcribe_stream(
        audio_path,
        max_chunk_seconds=30.0,           # ← tune if needed
        chunk_overlap_seconds=10.0
    ):
        partial_jp = partial_jp.strip()

        # Skip if no meaningful new content
        if not partial_jp or len(partial_jp) <= len(full_japanese):
            sys.stdout.write(f"\r{progress:>6.1%}  (processing...)")
            sys.stdout.flush()
            continue

        # ── New Japanese content arrived ───────────────────────────────
        new_jp_part = partial_jp[len(full_japanese):].lstrip()

        if new_jp_part:
            # IMPORTANT: Unpack only the translated text (first return value)
            new_en_part, *_ = translate_japanese_to_english(new_jp_part)
            #                                    ^^^   ignore logprob, confidence, label

            # Append new parts
            full_japanese = partial_jp
            if new_en_part.strip():
                full_english += (" " if full_english else "") + new_en_part.strip()

        # ── Simple overwriting line ───────────────────────────────
        jp_line = f"Japanese: {full_japanese or '(waiting for first transcription...)'}"
        en_line = f"English : {full_english or '(translation will appear here)'}"

        sys.stdout.write(f"\r{progress:>6.1%}  {jp_line[:120]}")
        sys.stdout.write("\n" + " " * 8 + en_line[:120])
        sys.stdout.flush()

    print("\n" + "=" * 80)
    print("✓ Streaming & Translation complete")
    print("=" * 80)

    print("\nFinal Result:")
    print("-" * 80)
    print("Japanese:")
    print(full_japanese)
    print()
    print("English:")
    print(full_english)
    print("-" * 80)


if __name__ == "__main__":
    AUDIO_SHORT = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
    AUDIO_LONG  = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

    print("\nDemo: Long Audio – Streaming / Live mode")
    live_stream_example(AUDIO_LONG)

    # print("Demo: Short Audio (normal mode)")
    # example(AUDIO_SHORT)

    # print("Demo: Long Audio (normal mode)")
    # example(AUDIO_LONG)
