from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, TypedDict

import jiwer
import pandas as pd
import torch
from typing import Literal
from datasets import Dataset
from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
import time

console = Console()
log = console.print  # easy alias for log output


def auto_detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_best_compute_config() -> dict[str, str | torch.dtype | Literal["cuda", "cpu"]]:
    """
    Detect hardware and return optimal Whisper inference settings.
    Supports your GTX 1660 (Turing architecture → fp16 supported).
    """
    log("[bold blue]Detecting hardware & best compute configuration...[/]")

    device: Literal["cuda", "cpu"] = "cpu"
    dtype = torch.float32
    compute_type = "float32"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        log(f"CUDA available → [green]{device_name}[/] (Compute Capability {capability[0]}.{capability[1]})")

        # GTX 1660 = Turing = sm_75 → full fp16 support
        if capability >= (7, 0):  # Turing and newer
            device = "cuda"
            dtype = torch.float16
            compute_type = "float16"
            log("[bold green]→ Using FP16 (fast & memory-efficient) on GTX 1660[/]")
        else:
            device = "cuda"
            log("[yellow]→ Falling back to FP32 on CUDA (older arch)[/]")
    else:
        log("[yellow]CUDA not available → using CPU (slow but safe)[/]")

    config = {
        "device": device,
        "dtype": dtype,
        "compute_type": compute_type,
    }
    log(f"Final inference config: [cyan]{config}[/]")
    return config


def _is_na(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and torch.isnan(torch.tensor(val)):
        return True
    try:
        return bool(pd.isna(val))
    except:
        return False


class ReferenceColumns(TypedDict):
    en: str | None  # Only English reference needed


class JapaneseS2TEvaluator:
    """
    Evaluator for Japanese Speech-to-Text Translation (audio → English text) using faster-whisper.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        compute_type: str | None = None,
        output_dir: Path | str | None = None,
        save_audio: bool = True,
        device: Literal["cuda", "cpu"] | None = None,  # allow passing deteced device
    ) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_audio = save_audio

        # Device auto-detection: honor supplied device or fallback
        self.device = device or auto_detect_device()
        self.compute_type = compute_type or get_best_compute_config()

        with console.status(f"[bold green]Loading faster-whisper {self.model_size}..."):
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        console.print(f"Loaded [bold cyan]{self.model_size}[/] → [bold green]{self.compute_type}[/] on [bold yellow]{self.device.upper()}[/]")

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for sub in ["audio", "en", "en_pred"]:
                (self.output_dir / sub).mkdir(exist_ok=True)
            console.print(f"All results will be saved to → [bold blue]{self.output_dir.resolve()}[/]")

    @staticmethod
    def _detect_reference_column(
        data: dict[str, list],
        text_columns: list[str],
    ) -> str | None:
        """Detect English reference column (most likely to be translation target)"""
        MIN_CONFIDENCE = 0.70
        MAX_SAMPLES = 50

        best_col = None
        best_score = 0

        for col in text_columns:
            en_count = 0
            checked = 0
            for val in data[col][:MAX_SAMPLES]:
                if _is_na(val) or not str(val).strip():
                    continue
                # Simple heuristic: English has more spaces and Latin chars
                text = str(val)
                if len(text.split()) > 3 and any(c.isalpha() for c in text):
                    en_count += 1
                checked += 1
            if checked > 0 and (en_count / checked) > 0.7 and en_count > best_score:
                best_score = en_count
                best_col = col

        return best_col or text_columns[0] if text_columns else None

    @staticmethod
    def from_extracted_json(
        json_path: Path | str,
        audio_dir: Path | str | None = None,
        reference_en_col: str | None = None,
        max_samples: int | None = None,
    ) -> list[dict]:
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"data.json not found: {json_path}")

        audio_dir = Path(audio_dir) if audio_dir else json_path.parent / "audio"
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        console.print(f"[bold blue]Loading S2T evaluation data:[/] {json_path.name}")
        data = json.loads(json_path.read_text(encoding="utf-8"))

        if "audio" not in data or not isinstance(data["audio"], list):
            raise ValueError("data.json must contain 'audio' list with filenames")

        text_cols = [k for k in data.keys() if k != "audio"]
        if not text_cols:
            raise ValueError("No text columns found in data.json")

        ref_en_col = reference_en_col or JapaneseS2TEvaluator._detect_reference_column(data, text_cols)
        if ref_en_col not in data:
            raise KeyError(f"English reference column '{ref_en_col}' not found. Available: {text_cols}")

        console.print(f"[bold green]Using English GT column:[/] [cyan]{ref_en_col}[/]")
        console.print(f"[green]Audio directory:[/] {audio_dir.resolve()}")

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
                "reference_en": str(data[ref_en_col][i]).strip(),
                "file_name": Path(audio_fn).name,
            }
            if sample["reference_en"]:
                samples.append(sample)

        console.print(f"[bold green]Loaded {len(samples)} valid S2T samples[/]")
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
            if ref_text:
                data.append({
                    "audio": str(wav_path),
                    "reference_en": ref_text,
                    "file_name": wav_path.name,
                })
            if max_samples and len(data) >= max_samples:
                break
        console.print(f"Loaded [bold blue]{len(data)}[/] audio-English text pairs")
        return Dataset.from_list(data)

    def translate_one(self, audio_input: str | Path) -> str:
        vad_params = {
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 400,
        }

        segments, _ = self.model.transcribe(
            str(audio_input),
            task="translate",  # ja → en
            beam_size=5,
            best_of=5,
            patience=2,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=vad_params,
        )

        text = "".join(seg.text.strip() for seg in segments).strip()
        return text if text else "[no speech detected]"

    def evaluate(
        self,
        samples: Dataset | list[dict],
        output_dir: Path | str | None = None,
        save_audio: bool | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        output_dir = Path(output_dir) if output_dir else self.output_dir
        save_audio = save_audio if save_audio is not None else self.save_audio

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for sub in ["audio", "en", "en_pred"]:
                (output_dir / sub).mkdir(exist_ok=True)

        console.rule("[bold magenta]Japanese → English S2T Translation Evaluation[/]")
        console.print(f"Processing [bold]{len(samples)}[/] samples")

        if isinstance(samples, Dataset):
            items = enumerate(samples)
            get_audio = lambda ex: ex["audio"]["path"] if isinstance(ex["audio"], dict) and "path" in ex["audio"] else ex["audio"]
            get_en = lambda ex: ex["reference_en"]
            get_name = lambda ex, i: ex.get("file_name", f"sample_{i:04d}")
        else:
            items = enumerate(samples)
            get_audio = lambda ex: ex["audio"]
            get_en = lambda ex: ex["reference_en"]
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
            prog_task = progress.add_task("Translating...", total=len(samples), speed=0.0)
            for i, ex in items:
                audio_input = get_audio(ex)
                ref_en = get_en(ex)
                filename = get_name(ex, i)

                pred_en = self.translate_one(audio_input)

                # Normalize for fair comparison
                ref_clean = " ".join(ref_en.lower().split())
                pred_clean = " ".join(pred_en.lower().split())

                wer = jiwer.wer(ref_clean, pred_clean) if ref_clean and pred_clean else float("nan")
                cer = jiwer.cer(ref_clean, pred_clean) if ref_clean and pred_clean else float("nan")

                if output_dir:
                    if save_audio and Path(audio_input).exists():
                        shutil.copy2(audio_input, output_dir / "audio" / Path(audio_input).name)
                    (output_dir / "en" / f"{filename}.txt").write_text(ref_en, encoding="utf-8")
                    (output_dir / "en_pred" / f"{filename}.txt").write_text(pred_en, encoding="utf-8")

                results.append({
                    "file": filename,
                    "reference_en": ref_en,
                    "prediction_en": pred_en,
                    "wer": wer,
                    "cer": cer,
                })

                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                progress.update(prog_task, advance=1, description=f"[cyan]{filename}[/]", speed=speed)

        df = pd.DataFrame(results)
        valid = df["wer"].notna()
        wer_mean = float(df.loc[valid, "wer"].mean())
        cer_mean = float(df.loc[valid, "cer"].mean())
        samples_count = int(valid.sum())

        if output_dir:
            df.to_csv(output_dir / "results.csv", index=False)
            summary = {
                "model": self.model_size,
                "task": "translate",
                "samples": len(df),
                "s2t": {"wer": wer_mean, "cer": cer_mean, "samples": samples_count},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        table = Table(title="Japanese → English S2T Evaluation Results")
        table.add_column("Task", style="bold cyan")
        table.add_column("WER", style="red")
        table.add_column("CER", style="yellow")
        table.add_column("Samples", style="green")

        wer_str = f"[bold green]{wer_mean:.2%}[/]" if wer_mean < 0.15 else f"[yellow]{wer_mean:.2%}[/]" if wer_mean < 0.30 else f"[red]{wer_mean:.2%}[/]"
        table.add_row("S2T (ja→en)", wer_str, f"{cer_mean:.2%}", str(samples_count))
        console.print(table)

        return df, {"s2t": {"wer": wer_mean, "cer": cer_mean, "samples": samples_count}}
