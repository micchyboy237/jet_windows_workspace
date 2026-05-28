from __future__ import annotations
import os
import re
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import (
    rich_transcription_postprocess,
    lang_dict,
    emo_dict,
    event_dict,
)
from sentence_utils import SYMBOL_RANGE, split_sentences_ja

TimestampPair = Tuple[int, int]

class WordSegment(TypedDict):
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    word: str

class PhraseSegment(TypedDict):
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    phrase: str
    word_segments: List[WordSegment]

class TranscriptionMetadata(TypedDict, total=False):
    processing_duration_sec: float
    model: str
    audio_duration_sec: float
    transcribed_duration_sec: float
    transcribed_duration_pctg: float
    coverage_label: str
    language: Optional[str]          # NEW: detected language (auto, zh, en, ja, ko, yue)
    emo: Optional[str]               # NEW: dominant emotion (HAPPY, SAD, ANGRY, etc.)
    event: Optional[str]             # NEW: detected events (BGM, Applause, Laughter, etc.)

class TranscriptionResult(TypedDict):
    text: str
    confidence: float
    quality_label: str
    avg_logprob: Optional[float]
    word_segments: list[WordSegment]
    phrase_segments: list[PhraseSegment]
    metadata: TranscriptionMetadata

if torch.cuda.is_available():
    device = "cuda:0"
    print("✅ CUDA is available. Using NVIDIA GPU (cuda:0)")
else:
    device = "cpu"
    print("⚠️  No GPU detected (neither CUDA nor MPS). Falling back to CPU")
print(f"Final selected device: {device}\n")

model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    disable_update=True,
    device=device,
    hub="hf",
)
print(f"Model successfully loaded on: {device}")

if device == "mps":
    print("🎉 Metal GPU acceleration (MPS) is active — good for your Mac M1!")
elif device == "cuda:0":
    print("🎉 CUDA GPU acceleration is active.")
else:
    print("Running on CPU (slower but always stable).")

SYMBOL_PATTERN = re.compile(rf"[{SYMBOL_RANGE}]+")
MAX_WORD_DURATION_SEC = 1.0
MIN_WORD_DURATION_SEC = 0.02
PUNCTUATION_SET = set("。、！？.,!?")

def extract_rich_metadata(raw_text: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Extract rich metadata from raw SenseVoiceSmall output.
    
    Parses special tokens like <|zh|>, <|HAPPY|>, <|BGM|> from the raw ASR text
    and returns the cleaned text along with detected language, emotion, and events.
    
    Args:
        raw_text: Raw ASR text containing special tokens
    
    Returns:
        Tuple of (cleaned_text, language, emotion, event)
        
    Example:
        >>> raw = "<|zh|><|HAPPY|><|BGM|>Hello world"
        >>> text, lang, emo, evt = extract_rich_metadata(raw)
        >>> print(text)  # "😊🎼Hello world"
        >>> print(lang)  # "zh"
        >>> print(emo)   # "HAPPY"
        >>> print(evt)   # "BGM"
    """
    # --- Extract Language ---
    # lang_dict keys are like "<|zh|>", "<|en|>", etc.
    language = None
    for lang_token in lang_dict:
        if lang_token in raw_text:
            # Extract language code: "<|zh|>" -> "zh"
            language = lang_token.strip("<|").strip("|>")
            break
    
    # --- Extract Emotion ---
    # emo_dict keys are like "<|HAPPY|>", "<|SAD|>", etc.
    # Pick the dominant emotion (highest count)
    emotion_counts = {}
    for emo_token in emo_dict:
        count = raw_text.count(emo_token)
        if count > 0:
            emotion_counts[emo_token] = count
    
    emotion = None
    if emotion_counts:
        # Get the emotion token with highest frequency
        dominant_emo_token = max(emotion_counts, key=emotion_counts.get)
        # "<|HAPPY|>" -> "HAPPY"
        emotion = dominant_emo_token.strip("<|").strip("|>")
    
    # --- Extract Events ---
    # event_dict keys are like "<|BGM|>", "<|Applause|>", etc.
    events_found = []
    for event_token in event_dict:
        if event_token in raw_text:
            event_name = event_token.strip("<|").strip("|>")
            events_found.append(event_name)
    
    # Join multiple events with comma
    event = ", ".join(events_found) if events_found else None
    
    # --- Clean Text ---
    # Use existing postprocessing to convert tokens to emojis
    cleaned_text = rich_transcription_postprocess(raw_text)
    
    return cleaned_text, language, emotion, event

def _postprocess_word_timing(
    word: Optional[str],
    start_sec: Optional[float],
    end_sec: Optional[float],
    duration_sec: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Normalize word-level timestamps from SenseVoiceSmall.
    Fixes:
    - punctuation inheriting silence
    - overly long durations from chunk alignment
    - near-zero durations
    """
    if word and word in PUNCTUATION_SET:
        duration_sec = MIN_WORD_DURATION_SEC
        if start_sec is not None:
            end_sec = start_sec + duration_sec
    if duration_sec is not None:
        if duration_sec > MAX_WORD_DURATION_SEC:
            duration_sec = MAX_WORD_DURATION_SEC
            if start_sec is not None:
                end_sec = start_sec + duration_sec
        elif duration_sec < MIN_WORD_DURATION_SEC:
            duration_sec = MIN_WORD_DURATION_SEC
            if start_sec is not None:
                end_sec = start_sec + duration_sec
    return start_sec, end_sec, duration_sec

def _normalize_phrase(text: str) -> str:
    """
    Normalize phrase text for alignment:
    - remove spaces
    - remove symbols (emoji, etc.)
    """
    text = re.sub(r"\s+", "", text)
    text = SYMBOL_PATTERN.sub("", text)
    return text

def _normalize_word(text: str) -> str:
    """
    Normalize ASR word token.
    """
    return re.sub(r"\s+", "", text or "")

def _build_phrase_segments(
    phrases: List[str],
    segments: List[WordSegment],
) -> List[PhraseSegment]:
    """
    Align phrases to word segments using sequential matching.
    Handles:
    - character-level ASR tokens
    - missing symbols in ASR
    """
    phrase_segments: List[PhraseSegment] = []
    seg_idx = 0
    total_segments = len(segments)
    for p_idx, phrase in enumerate(phrases):
        cleaned_phrase = phrase.strip()
        if not cleaned_phrase:
            continue
        normalized_phrase = _normalize_phrase(cleaned_phrase)
        phrase_len = len(normalized_phrase)
        if phrase_len == 0:
            continue
        collected: List[WordSegment] = []
        matched_chars = ""
        while seg_idx < total_segments and len(matched_chars) < phrase_len:
            seg = segments[seg_idx]
            word = seg.get("word") or ""
            normalized_word = _normalize_word(word)
            matched_chars += normalized_word
            collected.append(seg)
            seg_idx += 1
        if len(matched_chars) > phrase_len:
            overflow = len(matched_chars) - phrase_len
            while overflow > 0 and collected:
                last = collected[-1]
                last_word = _normalize_word(last.get("word") or "")
                if len(last_word) <= overflow:
                    overflow -= len(last_word)
                    collected.pop()
                else:
                    break
        if collected:
            start_sec = collected[0]["start_sec"]
            end_sec = collected[-1]["end_sec"]
            duration_sec = (
                (end_sec - start_sec)
                if start_sec is not None and end_sec is not None
                else 0.0
            )
            phrase_segments.append(
                {
                    "index": p_idx,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": round(duration_sec, 3),
                    "phrase": cleaned_phrase,
                    "word_segments": collected,
                }
            )
    return phrase_segments

def calculate_transcribed_duration_percentage(
    segments: list[WordSegment], total_audio_duration_seconds: float
) -> float:
    if total_audio_duration_seconds <= 0:
        return 0.0
    total_transcribed_sec = sum(seg.get("duration_sec") or 0.0 for seg in segments)
    percentage = (total_transcribed_sec / total_audio_duration_seconds) * 100
    return percentage

def get_audio_duration_seconds(audio_path: Path) -> float:
    sample_rate, data = wavfile.read(str(audio_path))
    return len(data) / sample_rate

def _transcribe_file(
    audio_path: Path,
    *,
    language: str = "auto",
    hotwords: str | list[str] | None = None,
) -> List[Dict[str, Any]]:
    """Transcribe with MPS OOM protection and automatic CPU fallback."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    try:
        with torch.inference_mode():
            results = model.generate(
                input=str(audio_path),
                cache=None,
                language=language,
                use_itn=True,
                batch_size=1,
                output_timestamp=True,
                hotwords=hotwords,
                merge_vad=False,
            )
        return results
    except Exception as e:
        error_str = str(e).lower()
        if device == "mps" and ("mps backend out of memory" in error_str or "out of memory" in error_str):
            from rich.console import Console
            console = Console()
            console.print("[bold red]MPS Out of Memory detected during CTC alignment.[/bold red]")
            console.print("[yellow]Retrying automatically on CPU...[/yellow]")
            torch.mps.empty_cache()
            try:
                with torch.inference_mode():
                    results = model.generate(
                        input=str(audio_path),
                        cache=None,
                        language="ja",
                        use_itn=True,
                        batch_size=1,
                        output_timestamp=True,
                        hotwords=hotwords,
                        merge_vad=False,
                    )
                console.print("[green]✅ Retry on CPU succeeded.[/green]")
                return results
            except Exception as fallback_e:
                console.print(f"[bold red]CPU fallback failed:[/bold red] {fallback_e}")
        from rich.console import Console
        console = Console()
        console.print(f"[bold red]Transcription error on {device}:[/bold red] {e}")
        console.print("[dim]Full traceback:[/dim]")
        console.print(traceback.format_exc())
        return []

def get_coverage_quality_label(pct: float) -> str:
    if pct >= 92.0: return "excellent (very clean)"
    if pct >= 82.0: return "good"
    if pct >= 65.0: return "fair (some noise/BGM)"
    if pct >= 40.0: return "sparse speech"
    if pct >= 15.0: return "very sparse"
    return "almost no speech"

def transcribe_audio_llm_from_file(
    audio_path: Path,
    *,
    language: str = "auto",
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
) -> TranscriptionResult:
    started = datetime.now(timezone.utc)
    raw_results = _transcribe_file(
        audio_path,
        language=language,
        hotwords=hotwords,
    )
    print(f"raw_results:\n{json.dumps(raw_results, ensure_ascii=False)}")
    
    if not raw_results:
        return TranscriptionResult(
            text="",
            language=None,
            emo=None,
            event=None,
            confidence=None,
            quality_label="N/A",
            avg_logprob=None,
            word_segments=[],
            phrase_segments=[],
            metadata={},
        )
    
    first = raw_results[0]
    raw_text = first["text"]
    
    # ✅ NEW: Extract rich metadata (language, emotion, events) and clean text
    ja_text, detected_language, detected_emotion, detected_event = extract_rich_metadata(raw_text)
    
    segments: list[WordSegment] = []
    timestamps = first.get("timestamp", [])
    words = first.get("words", [])
    for idx, ts in enumerate(timestamps):
        start_sec = end_sec = duration_sec = None
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start_ms, end_ms = ts[0], ts[1]
            if start_ms is not None:
                start_sec = start_ms / 1000.0
            if end_ms is not None:
                end_sec = end_ms / 1000.0
            if start_sec is not None and end_sec is not None:
                duration_sec = end_sec - start_sec
        word = words[idx] if idx < len(words) else None
        start_sec, end_sec, duration_sec = _postprocess_word_timing(
            word,
            start_sec,
            end_sec,
            duration_sec,
        )
        segments.append(
            {
                "index": idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round(duration_sec, 3)
                if duration_sec is not None
                else None,
                "word": word,
            }
        )
    
    audio_duration = get_audio_duration_seconds(audio_path)
    transcribed_percentage = calculate_transcribed_duration_percentage(
        segments, audio_duration
    )
    transcribed_duration_sec = (transcribed_percentage / 100) * audio_duration
    coverage_label = get_coverage_quality_label(transcribed_percentage)
    
    metadata: TranscriptionMetadata = {
        "model": "SenseVoiceSmall",
        "processing_duration_sec": round(
            (datetime.now(timezone.utc) - started).total_seconds(), 3
        ),
        "audio_duration_sec": round(audio_duration, 3),
        "transcribed_duration_sec": round(transcribed_duration_sec, 3),
        "transcribed_duration_pctg": round(transcribed_percentage, 2),
        "coverage_label": coverage_label,
        "language": detected_language,   # ✅ NEW
        "emo": detected_emotion,         # ✅ NEW
        "event": detected_event,         # ✅ NEW
    }
    
    phrase_segments: list[PhraseSegment] = []
    if segments and ja_text.strip():
        ja_text = ja_text.replace(" ", '、')
        phrases = split_sentences_ja(ja_text)
        if ja_text and re.fullmatch(rf"[{re.escape('。、！？.,!? \n\t')}]+\s*", ja_text):
            ja_text = ""
        phrase_segments = _build_phrase_segments(phrases, segments)
        ja_text = "".join(phrases)
    
    return {
        "text": ja_text,
        "confidence": None,
        "quality_label": None,
        "avg_logprob": None,
        "word_segments": segments,
        "phrase_segments": phrase_segments,
        "metadata": metadata,
    }

def transcribe_audio(
    audio_bytes: bytes,
    sample_rate: int,
    *,
    language: str = "auto",
    hotwords: str | list[str] | None = None,
    context_prompt: str | None = None,
    save_temp_wav: Path | None = None,
) -> TranscriptionResult:
    """
    Transcribe raw PCM int16 bytes.
    Designed for live server usage.
    """
    processing_started = datetime.now(timezone.utc)
    if save_temp_wav:
        audio_path = save_temp_wav
        audio_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = Path(tmp.name)
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    wavfile.write(str(audio_path), sample_rate, arr)
    result = transcribe_audio_llm_from_file(
        audio_path,
        language=language,
        hotwords=hotwords,
        context_prompt=context_prompt,
    )
    if not save_temp_wav:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
    return result

if __name__ == "__main__":
    import argparse
    import json
    import shutil
    from pathlib import Path
    from rich.console import Console
    from rich.pretty import pprint
    from scipy.io import wavfile
    from translate_jp_en_llm import translate_japanese_to_english
    
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    default_audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\start_15s_recording_1_speaker.wav"
    
    parser = argparse.ArgumentParser(description="ASR demo with language detection.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=default_audio_path,
        help="Audio to transcribe (optional, defaults to sample audio path)",
    )
    parser.add_argument(
        "-l", "--language",
        default="auto",
        help="Language code, e.g. 'ja', 'en', 'auto' (default).",
    )
    args = parser.parse_args()
    audio_path = Path(args.audio_path)
    language = args.language
    
    console = Console()
    
    result: TranscriptionResult = transcribe_audio_llm_from_file(
        audio_path,
        language=language,
    )
    
    # ✅ Extract fields
    transcribed_text = result.pop("text")
    detected_language = result.pop("language")
    detected_emotion = result.pop("emo")
    detected_event = result.pop("event")
    word_segments = result.pop("word_segments")
    phrase_segments = result.pop("phrase_segments")
    metadata = result.pop("metadata")
    
    # ✅ Display rich metadata
    console.print(f"[cyan]🔤 Detected Language:[/cyan] {detected_language or 'unknown'}")
    console.print(f"[cyan]😊 Detected Emotion:[/cyan] {detected_emotion or 'neutral'}")
    console.print(f"[cyan]🎵 Detected Events:[/cyan] {detected_event or 'none'}")
    
    # ✅ Language-specific processing
    language_map = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "yue": "Cantonese",
    }
    detected_lang_name = language_map.get(detected_language, detected_language or "Unknown")
    console.print(f"[bold green]{detected_lang_name}:[/bold green]")
    
    # ✅ Only apply Japanese-specific processing if language is Japanese
    if detected_language == "ja":
        # Japanese-specific processing already done in transcribe_audio_llm_from_file
        console.print(f"JA:\n[bold cyan]{transcribed_text}[/bold cyan]")
        
        # ✅ Translate only if Japanese
        console.print("[dim]Loading translator...[/dim]")
        try:
            en_text = translate_japanese_to_english(transcribed_text)["text"]
            console.print(f"EN:\n[bold cyan]{en_text}[/bold cyan]")
            
            # Save English translation
            en_text_path = OUTPUT_DIR / "en_text.md"
            with open(en_text_path, "w", encoding="utf-8") as f:
                f.write(en_text)
            console.print(
                f"[bold green]Saved en_text to:[/bold green] [link=file://{en_text_path.resolve()}]{en_text_path}[/link]"
            )
        except Exception as e:
            console.print(f"[yellow]Translation failed: {e}[/yellow]")
    else:
        # ✅ For non-Japanese languages, just display the transcribed text
        console.print(f"Text:\n[bold cyan]{transcribed_text}[/bold cyan]")
    
    pprint(result, expand_all=True)
    
    # Save all outputs
    scores_json_path = OUTPUT_DIR / "scores.json"
    with open(scores_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved scores to:[/bold green] [link=file://{scores_json_path.resolve()}]{scores_json_path}[/link]"
    )
    
    # ✅ Save rich metadata
    rich_meta_path = OUTPUT_DIR / "rich_metadata.json"
    rich_meta = {
        "language": detected_language,
        "language_name": detected_lang_name,
        "emotion": detected_emotion,
        "event": detected_event,
    }
    with open(rich_meta_path, "w", encoding="utf-8") as f:
        json.dump(rich_meta, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved rich_metadata to:[/bold green] [link=file://{rich_meta_path.resolve()}]{rich_meta_path}[/link]"
    )
    
    # ✅ Save transcribed text (language-agnostic name)
    text_path = OUTPUT_DIR / "transcribed_text.md"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(transcribed_text)
    console.print(
        f"[bold green]Saved transcribed_text to:[/bold green] [link=file://{text_path.resolve()}]{text_path}[/link]"
    )
    
    word_segments_path = OUTPUT_DIR / "word_segments.json"
    with open(word_segments_path, "w", encoding="utf-8") as f:
        json.dump(word_segments, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved word_segments to:[/bold green] [link=file://{word_segments_path.resolve()}]{word_segments_path}[/link]"
    )
    
    phrase_segments_path = OUTPUT_DIR / "phrase_segments.json"
    with open(phrase_segments_path, "w", encoding="utf-8") as f:
        json.dump(phrase_segments, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved phrase_segments to:[/bold green] [link=file://{phrase_segments_path.resolve()}]{phrase_segments_path}[/link]"
    )
    
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]Saved metadata to:[/bold green] [link=file://{metadata_path.resolve()}]{metadata_path}[/link]"
    )
    
    # ✅ Only create phrase segments if they exist (works for all languages)
    if phrase_segments:
        phrases_dir = OUTPUT_DIR / "phrases"
        phrases_dir.mkdir(parents=True, exist_ok=True)
        console.print(
            f"[bold green]Created phrases directory:[/bold green] [link=file://{phrases_dir.resolve()}]{phrases_dir}[/link]"
        )
        
        sample_rate, full_audio_data = wavfile.read(str(audio_path))
        for phrase in phrase_segments:
            phrase_num = phrase["index"]
            phrase_dir = phrases_dir / f"phrase_{phrase_num}"
            phrase_dir.mkdir(parents=True, exist_ok=True)
            meta_path = phrase_dir / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(phrase, f, ensure_ascii=False, indent=2)
            console.print(
                f"[bold green]Saved meta.json to:[/bold green] [link=file://{meta_path.resolve()}]{meta_path}[/link]"
            )
            start_sec = phrase.get("start_sec")
            end_sec = phrase.get("end_sec")
            if start_sec is not None and end_sec is not None:
                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate)
                sliced_data = full_audio_data[start_sample:end_sample]
                sound_path = phrase_dir / "sound.wav"
                wavfile.write(str(sound_path), sample_rate, sliced_data)
                console.print(
                    f"[bold green]Saved sound.wav to:[/bold green] [link=file://{sound_path.resolve()}]{sound_path}[/link]"
                )
            else:
                console.print(
                    f"[yellow]Skipping sound.wav for phrase_{phrase_num} (no timestamps)[/yellow]"
                )
    
    console.print(
        "[bold green]✅ Per-phrase audio + meta export complete![/bold green]"
    )
