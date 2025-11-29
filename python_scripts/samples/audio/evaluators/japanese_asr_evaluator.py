from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal, TypedDict

import jiwer
import pandas as pd
import torch
import unicodedata
from datasets import Dataset
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
import time

from jet.utils.language import detect_lang

console = Console()


def normalize_japanese_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = jiwer.RemovePunctuation()(text)
    text = jiwer.RemoveMultipleSpaces()(text)
    text = jiwer.RemoveWhiteSpace()(text)
    return text.strip()


def auto_detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_na(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (float,)) and torch.isnan(torch.tensor(val)):
        return True
    try:
        return bool(pd.isna(val))
    except:
        return False


class ReferenceColumns(TypedDict):
    ja: str
    en: str | None


class JapaneseASREvaluator:
    """
    Generic Japanese Speech-to-Text (S2T) evaluator with bilingual support.
    Supports 'translate' or 'both' (default) tasks.
    """

    TaskType = Literal["transcribe", "translate", "both"]

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str | None = None,
        output_dir: Path | str | None = None,
        save_audio: bool = True,
        task: str | None = None,  # Now optional → defaults to "both"
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_audio = save_audio

        # Default to evaluating both tasks
        self.task: JapaneseASREvaluator.TaskType = (task or "both").lower()  # type: ignore
        if self.task not in {"transcribe", "translate", "both"}:
            raise ValueError("task must be 'transcribe', 'translate', or 'both'")

        self.device = auto_detect_device()
        default_ct = "float16" if self.device == "cuda" else "int8"
        ct = self.compute_type or default_ct

        with console.status(f"[bold green]Loading faster-whisper {self.model_size}..."):
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=ct,
            )

        console.print(f"Loaded [bold cyan]{self.model_size}[/] → [bold green]{ct}[/] on [bold yellow]{self.device.upper()}[/]")

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for sub in ["audio", "ja", "ja_pred", "en", "en_pred"]:
                (self.output_dir / sub).mkdir(exist_ok=True)
            console.print(f"All results will be saved to → [bold blue]{self.output_dir.resolve()}[/]")

    @staticmethod
    def _detect_reference_columns(
        data: dict[str, list],
        text_columns: list[str],
    ) -> ReferenceColumns:
        """
        Detect best Japanese and English reference columns using language detection.
        Returns both, with en possibly None if not found.
        """
        MIN_CONFIDENCE = 0.70
        MAX_SAMPLES_PER_COL = 50
        MIN_REQUIRED_VOTES = 5  # Safety threshold

        col_scores: dict[str, dict[str, int]] = {}

        for col in text_columns:
            ja_votes = en_votes = checked = 0
            for val in data[col]:
                if checked >= MAX_SAMPLES_PER_COL:
                    break
                if _is_na(val):
                    continue
                text = str(val).strip()
                if not text:
                    continue
                detection = detect_lang(text)
                if detection["score"] >= MIN_CONFIDENCE:
                    if detection["lang"] == "ja":
                        ja_votes += 1
                    elif detection["lang"] == "en":
                        en_votes += 1
                checked += 1
            col_scores[col] = {"ja": ja_votes, "en": en_votes, "checked": checked}

        # Find best columns
        best_ja_col = max(col_scores.items(), key=lambda x: x[1]["ja"], default=(text_columns[0], {"ja": 0}))[0]
        best_en_col_entry = max(col_scores.items(), key=lambda x: x[1]["en"], default=(None, {"en": 0}))
        best_en_col = best_en_col_entry[0] if best_en_col_entry[1]["en"] > 0 else None

        # Log detailed results
        console.print("[green]Language detection results per column:[/]")
        for col, scores in sorted(col_scores.items(), key=lambda x: (x[1]["ja"] + x[1]["en"]), reverse=True):
            mark = ""
            if col == best_ja_col and col == best_en_col:
                mark = "both"
            elif col == best_ja_col:
                mark = "ja"
            elif col == best_en_col:
                mark = "en"
            console.print(f"  {mark:>4} → {col}: ja={scores['ja']}, en={scores['en']}, checked={scores['checked']}")

        result: ReferenceColumns = {"ja": best_ja_col, "en": best_en_col}

        # Validate required columns exist
        if col_scores[best_ja_col]["ja"] < MIN_REQUIRED_VOTES:
            raise ValueError(
                f"Could not reliably detect Japanese reference column. "
                f"Best candidate '{best_ja_col}' only got {col_scores[best_ja_col]['ja']} ja_votes "
                f"(need ≥{MIN_REQUIRED_VOTES}). Available columns: {text_columns}"
            )

        if best_en_col and col_scores[best_en_col]["en"] < MIN_REQUIRED_VOTES:
            console.print(f"[yellow]Warning: English reference weak ({col_scores[best_en_col]['en']} votes), will skip translation metrics[/]")
            result["en"] = None

        return result

    @staticmethod
    def from_extracted_json(
        json_path: Path | str,
        audio_dir: Path | str | None = None,
        reference_ja_col: str | None = None,
        reference_en_col: str | None = None,
        max_samples: int | None = None,
        task: str | None = None,
    ) -> list[dict]:
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"data.json not found: {json_path}")

        audio_dir = Path(audio_dir) if audio_dir else json_path.parent / "audio"
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        console.print(f"[bold blue]Loading extracted evaluation data:[/] {json_path.name}")
        data = json.loads(json_path.read_text(encoding="utf-8"))

        if "audio" not in data or not isinstance(data["audio"], list):
            raise ValueError("data.json must contain 'audio' list with filenames")

        text_cols = [k for k in data.keys() if k != "audio"]
        if not text_cols:
            raise ValueError("No text columns found in data.json")

        task = (task or "both").lower()
        if task not in {"transcribe", "translate", "both"}:
            raise ValueError("task must be 'transcribe', 'translate', or 'both'")

        # Detect or use manual reference columns
        if reference_ja_col or reference_en_col:
            if reference_ja_col and reference_ja_col not in data:
                raise KeyError(f"reference_ja_col '{reference_ja_col}' not found")
            if reference_en_col and reference_en_col not in data:
                raise KeyError(f"reference_en_col '{reference_en_col}' not found")
            ref_cols: ReferenceColumns = {
                "ja": reference_ja_col or text_cols[0],
                "en": reference_en_col if reference_en_col in data else None,
            }
            console.print(f"[green]Using manual reference columns → ja:[cyan]{ref_cols['ja']}[/] en:[cyan]{ref_cols['en'] or 'N/A'}[/]")
        else:
            ref_cols = JapaneseASREvaluator._detect_reference_columns(data, text_cols)

        # Enforce required columns based on task
        if task in {"transcribe", "both"} and ref_cols["ja"] not in data:
            raise ValueError(f"Japanese reference column '{ref_cols['ja']}' not found in data.json")
        if task in {"translate", "both"} and not ref_cols["en"]:
            raise ValueError("Task includes 'translate' but no English reference column was detected or provided")

        console.print(f"[bold green]Using →[/] Japanese GT: [cyan]{ref_cols['ja']}[/] | English GT: [cyan]{ref_cols['en'] or 'N/A'}[/]")
        console.print(f"[green]Audio directory:[/] {audio_dir.resolve()}")
        console.print(f"[green]Total entries:[/] {len(data['audio'])}")

        samples = []
        for i, audio_fn in enumerate(data["audio"]):
            if max_samples and len(samples) >= max_samples:
                break
            if not audio_fn or _is_na(audio_fn):
                continue

            audio_path = (audio_dir / audio_fn).resolve()
            if not audio_path.exists():
                console.print(f"[dim]Missing audio:[/] {audio_path.name} → skipping")
                continue

            sample = {
                "audio": str(audio_path),
                "reference_ja": str(data[ref_cols["ja"]][i]).strip(),
                "reference_en": str(data[ref_cols["en"]][i]).strip() if ref_cols["en"] else "",
                "file_name": Path(audio_fn).name,
            }
            samples.append(sample)

        console.print(f"[bold green]Loaded {len(samples)} valid samples[/]")
        return samples

    @staticmethod
    def from_folder(audio_dir: Path | str, text_dir: Path | str, max_samples: int | None = None) -> Dataset:
        audio_dir = Path(audio_dir)
        text_dir = Path(text_dir)
        data = []
        for wav_path in sorted(audio_dir.glob("*.wav")):
            txt_path = text_dir / (wav_path.stem + ".txt")
            if not txt_path.exists():
                continue
            ref_text = txt_path.read_text(encoding="utf-8").strip()
            data.append({
                "audio": str(wav_path),
                "reference_ja": ref_text,  # assume Japanese
                "reference_en": "",
                "file_name": wav_path.name,
            })
            if max_samples and len(data) >= max_samples:
                break
        console.print(f"Loaded [bold blue]{len(data)}[/] audio-text pairs")
        return Dataset.from_list(data)

    def transcribe_one(self, audio_input: str | Path, task: str = "transcribe") -> str:
        vad_params = {
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 2000,   # increased from 1000 → better segment boundary
            "speech_pad_ms": 400,
            "max_speech_duration_s": float("inf"),
        }

        # Critical fix: Use temperature 0.0 only for Japanese transcription
        # Translation is robust to higher temperatures, but ja→ja is NOT
        if task == "transcribe":
            temperatures = [0.0]                            # ← ONLY zero temp
            beam_size = 5
            best_of = 5
        else:  # translate
            temperatures = [0.0, 0.2, 0.4]                  # safe for translation
            beam_size = 5
            best_of = 5

        for temp in temperatures:
            segments, info = self.model.transcribe(
                str(audio_input),
                language="ja",
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=2,
                temperature=temp,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=True,
                vad_parameters=vad_params,
            )

            text = "".join(seg.text.strip() for seg in segments).strip()
            # For Japanese: preserve natural spacing (no forced spaces between segments)
            if task == "transcribe":
                text = text.replace(" ", "")  # Whisper sometimes adds spaces in ja mode

            if len(text) >= 10:  # lowered threshold slightly
                return text

        # Final fallback: return whatever we have
        return text if text else "[no speech detected]"

    def evaluate(
        self,
        samples: Dataset | list[dict],
        output_dir: Path | str | None = None,
        save_audio: bool | None = None,
        task: str | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        output_dir = Path(output_dir) if output_dir else self.output_dir
        save_audio = save_audio if save_audio is not None else self.save_audio
        task = (task or self.task)

        do_transcribe = task in {"transcribe", "both"}
        do_translate = task in {"translate", "both"}

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for sub in ["audio", "ja", "ja_pred", "en", "en_pred"]:  # ← Added "en_pred"
                (output_dir / sub).mkdir(exist_ok=True)

        console.rule(f"[bold magenta]Japanese S2T Evaluation — Task: {task.upper()}[/]")
        console.print(f"Processing [bold]{len(samples)}[/] samples")

        if isinstance(samples, Dataset):
            items = enumerate(samples)
            get_audio = lambda ex: ex["audio"]["path"] if isinstance(ex["audio"], dict) and "path" in ex["audio"] else ex["audio"]
            get_ja = lambda ex: ex.get("reference_ja") or ex["reference"]
            get_en = lambda ex: ex.get("reference_en", "")
            get_name = lambda ex, i: ex.get("file_name", f"sample_{i:04d}")
        else:
            items = enumerate(samples)
            get_audio = lambda ex: ex["audio"]
            get_ja = lambda ex: ex["reference_ja"]
            get_en = lambda ex: ex.get("reference_en", "")
            get_name = lambda ex, i: ex.get("file_name", f"sample_{i:04d}")

        results = []
        start_time = time.time()
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[speed]:.2f} files/s"),
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            prog_task = progress.add_task("Processing...", total=len(samples), speed=0.0)
            for i, ex in items:
                audio_input = get_audio(ex)
                ref_ja = get_ja(ex)
                ref_en = get_en(ex)
                filename = get_name(ex, i)

                pred_ja = self.transcribe_one(audio_input, task="transcribe") if do_transcribe else ""
                pred_en = self.transcribe_one(audio_input, task="translate") if do_translate else ""

                # Only evaluate S2T: Japanese audio → English translation
                asr_wer = asr_cer = float("nan")  # For compatibility, but not computed
                s2t_wer = s2t_cer = float("nan")

                if do_translate and ref_en.strip() and pred_en.strip():
                    # Simple English normalization
                    ref_clean = " ".join(ref_en.lower().split())
                    pred_clean = " ".join(pred_en.lower().split())
                    s2t_wer = jiwer.wer(ref_clean, pred_clean)
                    s2t_cer = jiwer.cer(ref_clean, pred_clean)

                if output_dir:
                    if save_audio and Path(audio_input).exists():
                        shutil.copy2(audio_input, output_dir / "audio" / Path(audio_input).name)
                    if ref_ja:
                        (output_dir / "ja" / f"{filename}.txt").write_text(ref_ja, encoding="utf-8")
                    if ref_en:
                        (output_dir / "en" / f"{filename}.txt").write_text(ref_en, encoding="utf-8")
                    if pred_ja:
                        (output_dir / "ja_pred" / f"{filename}.txt").write_text(pred_ja, encoding="utf-8")
                    if pred_en:
                        (output_dir / "en_pred" / f"{filename}.txt").write_text(pred_en, encoding="utf-8")

                # Store per-sample results
                results.append({
                    "file": filename,
                    "reference_ja": ref_ja,
                    "reference_en": ref_en or "",
                    "prediction_ja": pred_ja,
                    "prediction_en": pred_en,
                    "asr_wer": asr_wer,
                    "asr_cer": asr_cer,
                    "s2t_wer": s2t_wer,
                    "s2t_cer": s2t_cer,
                })

                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                progress.update(prog_task, advance=1, description=f"[cyan]{filename}[/]", speed=speed)

        # === Aggregation and output ===

        df = pd.DataFrame(results)

        # Only aggregate S2T metrics
        asr_wer = float("nan")
        asr_cer = float("nan")
        s2t_wer = float(df["s2t_wer"].mean())
        s2t_cer = float(df["s2t_cer"].mean())

        asr_samples = 0
        s2t_samples = int(len(df[df["s2t_wer"].notna()]))

        if output_dir:
            df.to_csv(output_dir / "results.csv", index=False)
            summary = {
                "model": self.model_size,
                "task": task,
                "samples": len(df),
                "asr": {"wer": asr_wer, "cer": asr_cer, "samples": asr_samples},
                "s2t": {"wer": s2t_wer, "cer": s2t_cer, "samples": s2t_samples},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            # Now safe to serialize
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )

        table = Table(title="Japanese S2T Evaluation Results")
        table.add_column("Task", style="bold cyan")
        table.add_column("WER", style="red")
        table.add_column("CER", style="yellow")
        table.add_column("Samples", style="green")

        # No ASR row
        if do_translate:
            if pd.isna(s2t_wer):
                wer_str = "N/A"
            elif s2t_wer < 0.05:
                wer_str = f"[bold green]{s2t_wer:.2%}[/]"
            elif s2t_wer < 0.15:
                wer_str = f"[green]{s2t_wer:.2%}[/]"
            else:
                wer_str = f"[yellow]{s2t_wer:.2%}[/]"
            table.add_row(
                "S2T (ja→en)",
                wer_str,
                f"{s2t_cer:.2%}" if not pd.isna(s2t_cer) else "N/A",
                str(s2t_samples)
            )

        console.print(table)

        return df, {
            "asr": {"wer": asr_wer, "cer": asr_cer, "samples": asr_samples},
            "s2t": {"wer": s2t_wer, "cer": s2t_cer, "samples": s2t_samples},
        }