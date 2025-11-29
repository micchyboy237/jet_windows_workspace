from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

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

from jet.utils.language import detect_lang  # Robust language detection

console = Console()


def normalize_japanese_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = jiwer.RemovePunctuation()(text)
    text = jiwer.RemoveMultipleSpaces()(text)
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


class JapaneseASREvaluator:
    """
    Generic Japanese ASR + Speech-to-Text (S2T) evaluator.
    Supports Hugging Face Datasets and extracted JSON + audio folder format.
    """

    TaskType = Literal["transcribe", "translate"]

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str | None = None,
        output_dir: Path | str | None = None,
        save_audio: bool = True,
        task: str = "translate",
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.output_dir = output_dir
        self.save_audio = save_audio
        self.task = (task or "translate").lower()
        if self.task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")

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
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for sub in ["audio", "ja", "ja_pred", "en"]:
                (self.output_dir / sub).mkdir(exist_ok=True)
            console.print(f"All results will be saved to → [bold blue]{self.output_dir.resolve()}[/]")

    @staticmethod
    def _auto_detect_reference_column(
        data: dict[str, list],
        text_columns: list[str],
        task: TaskType,
    ) -> str:
        """
        Robustly detect the ground truth reference column using fasttext-based language detection.
        Returns the column most likely to contain Japanese (transcribe) or English (translate) text.
        """
        MIN_CONFIDENCE = 0.70
        MAX_SAMPLES_PER_COL = 50

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

        # Select best column based on task
        if task == "transcribe":
            best_col = max(col_scores.items(), key=lambda x: x[1]["ja"], default=(text_columns[0], {"ja": 0}))[0]
            reason = f"ja_votes={col_scores[best_col]['ja']}"
        else:  # translate → expect English reference
            best_col = max(col_scores.items(), key=lambda x: x[1]["en"], default=(text_columns[0], {"en": 0}))[0]
            reason = f"en_votes={col_scores[best_col]['en']}"

        # Final conventional fallback
        final_col = "reference" if "reference" in text_columns else best_col

        console.print(
            f"[green]Auto-detected reference column:[/] [bold cyan]{final_col}[/] "
            f"({reason}, task={task})"
        )
        return final_col

    @staticmethod
    def from_extracted_json(
        json_path: Path | str,
        audio_dir: Path | str | None = None,
        reference_col: str | None = None,
        max_samples: int | None = None,
        task: str = "translate",
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

        task = (task or "translate").lower()
        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")

        # === Reference column selection (robust + reusable) ===
        if reference_col:
            if reference_col not in data:
                raise KeyError(f"Reference column '{reference_col}' not found. Available: {text_cols}")
            ref_col = reference_col
            console.print(f"[green]Using user-specified reference column:[/] [bold cyan]{ref_col}[/]")
        else:
            ref_col = JapaneseASREvaluator._auto_detect_reference_column(
                data=data,
                text_columns=text_cols,
                task=task,  # type: ignore
            )

        console.print(f"[bold green]Final ground truth column:[/] [bold cyan]{ref_col}[/] (task: {task})")
        console.print(f"[green]Audio directory:[/] {audio_dir.resolve()}")
        console.print(f"[green]Total entries in JSON:[/] {len(data['audio'])}")
        if max_samples is not None:
            console.print(f"[dim]Limiting to first [bold]{max_samples}[/] samples[/]")

        samples = []
        for i, (audio_fn, ref_text) in enumerate(zip(data["audio"], data[ref_col])):
            if max_samples is not None and len(samples) >= max_samples:
                console.print(f"[dim]Limit reached: stopping at {len(samples)} samples[/]")
                break
            if not audio_fn or _is_na(ref_text) or _is_na(audio_fn):
                continue

            audio_path = (audio_dir / audio_fn).resolve()
            if not audio_path.exists():
                console.print(f"[dim]Missing audio:[/] {audio_path.name} → skipping")
                continue

            samples.append({
                "audio": str(audio_path),
                "reference": str(ref_text).strip(),
                "file_name": Path(audio_fn).name,
            })

        console.print(f"[bold green]Successfully loaded {len(samples)} valid samples[/]")
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
                "reference": ref_text,
                "file_name": wav_path.name,
            })
            if max_samples and len(data) >= max_samples:
                break
        console.print(f"Loaded [bold blue]{len(data)}[/] audio-text pairs")
        return Dataset.from_list(data)

    def transcribe_one(self, audio_input: str | Path, task: str = "transcribe") -> str:
        segments, _ = self.model.transcribe(
            str(audio_input),
            language="ja",
            task=task,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())

    def evaluate(
        self,
        samples: Dataset | list[dict],
        output_dir: Path | str | None = None,
        save_audio: bool | None = None,
        task: str | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        output_dir = Path(output_dir) if output_dir else self.output_dir
        save_audio = save_audio if save_audio is not None else self.save_audio
        task = (task or self.task).lower()
        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")
        do_translate = task == "translate"

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for sub in ["audio", "ja", "ja_pred", "en"]:
                (output_dir / sub).mkdir(exist_ok=True)

        console.rule("[bold magenta]Japanese ASR + S2T Evaluation[/]")
        console.print(f"Processing [bold]{len(samples)}[/] samples — Task: [bold green]{task}[/]")

        if isinstance(samples, Dataset):
            items = enumerate(samples)
            get_audio = lambda ex: ex["audio"]["path"] if isinstance(ex["audio"], dict) and "path" in ex["audio"] else ex["audio"]
            get_ref = lambda ex: ex["reference"]
            get_name = lambda ex, i: ex.get("file_name", f"sample_{i:04d}")
        else:
            items = enumerate(samples)
            get_audio = lambda ex: ex["audio"]
            get_ref = lambda ex: ex["reference"]
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
                ref_ja = get_ref(ex)
                filename = get_name(ex, i)

                pred_ja = self.transcribe_one(audio_input, task="transcribe")
                pred_en = self.transcribe_one(audio_input, task="translate") if do_translate else ""

                if do_translate:
                    ref_n = normalize_japanese_text(pred_en)
                    pred_n = normalize_japanese_text(pred_en)
                else:
                    ref_n = normalize_japanese_text(ref_ja)
                    pred_n = normalize_japanese_text(pred_ja)

                wer = jiwer.wer(ref_n, pred_n)
                cer = jiwer.cer(ref_n, pred_n)

                if output_dir:
                    if save_audio and Path(audio_input).exists():
                        shutil.copy2(audio_input, output_dir / "audio" / Path(audio_input).name)
                    (output_dir / "ja" / f"{filename}.txt").write_text(ref_ja, encoding="utf-8")
                    (output_dir / "ja_pred" / f"{filename}.txt").write_text(pred_ja, encoding="utf-8")
                    if pred_en:
                        (output_dir / "en" / f"{filename}.txt").write_text(pred_en, encoding="utf-8")

                results.append({
                    "file": filename,
                    "reference_ja": ref_ja,
                    "prediction_ja": pred_ja,
                    "prediction_en": pred_en,
                    "wer": wer,
                    "cer": cer,
                })

                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                progress.update(prog_task, advance=1, description=f"[cyan]{filename}[/]", speed=speed)

        df = pd.DataFrame(results)
        total_wer = df["wer"].mean()
        total_cer = df["cer"].mean()

        if output_dir:
            df.to_csv(output_dir / "results.csv", index=False)
            summary = {
                "model": self.model_size,
                "device": self.device,
                "samples": len(df),
                "wer": total_wer,
                "cer": total_cer,
                "task": task,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        table = Table(title="Evaluation Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("WER", f"{total_wer:.2%}")
        table.add_row("CER", f"{total_cer:.2%}")
        table.add_row("Samples", str(len(df)))
        table.add_row("Task", task)
        console.print(table)

        return df, {"wer": total_wer, "cer": total_cer}