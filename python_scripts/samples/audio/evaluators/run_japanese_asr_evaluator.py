# Jet_Windows_Workspace/python_scripts/samples/audio/evaluators/run_japanese_asr_benchmarks.py
from __future__ import annotations

import shutil
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from japanese_asr_evaluator import JapaneseASREvaluator
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

console = Console()
log = console.log
status = console.status


class MetricsDict(TypedDict):
    wer: float
    cer: float


@dataclass(frozen=True)
class BenchmarkConfig:
    sample_limit: int = 1
    output_subdir: str = "whisper"


def get_project_root() -> Path:
    return Path(__file__).parent.resolve()


def make_output_dir() -> Path:
    output_dir = get_project_root() / "generated" / Path(__file__).stem
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log("[green]Clean output directory ready:[/] ", output_dir.resolve())
    return output_dir


def save_results(results_df: pd.DataFrame, metrics: MetricsDict, output_dir: Path, model_name: str) -> None:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{model_name}.csv"
    results_df.to_csv(csv_path, index=False)

    from rich.table import Table
    table = Table(title=f"[bold]Whisper Benchmark Result ({model_name.upper()})[/]", show_header=True, header_style="bold blue")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Interpretation")
    table.add_row("WER", f"[red]{metrics['wer']:.2%}[/]", "Lower = Better")
    table.add_row("CER", f"[yellow]{metrics['cer']:.2%}[/]", "Lower = Better")
    table.add_row("Samples", str(len(results_df)), "")
    table.add_row("Model", "large-v3", "float16")
    console.print(table)
    log(f"Full results → [link=file://{csv_path}]{csv_path}[/]")

    if metrics["wer"] < 0.12:
        rprint(Panel("[bold green]EXCELLENT![/] Top-tier result!", title="Achievement", style="green on black"))
    elif metrics["wer"] < 0.20:
        rprint(Panel("Solid baseline — close to published results", style="yellow"))
    else:
        rprint(Panel("Room for improvement — try fine-tuning!", style="red"))


def main(config: BenchmarkConfig | None = None) -> None:
    config = config or BenchmarkConfig()
    console.rule("[bold cyan]Japanese ASR Benchmark — Whisper large-v3[/]")
    log(f"Running with limit = [bold]{config.sample_limit}[/] samples")

    output_dir = make_output_dir()

    # Prefer offline extracted data.json
    default_json = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/python_scripts/samples/audio/generated/extract_parquet_data/data.json"
    if Path(default_json).exists():
        console.print("[bold green]Found extracted data → using offline mode[/]")
        samples = JapaneseASREvaluator.from_extracted_json(
            default_json,
            max_samples=config.sample_limit,   # ← This was missing!
        )
    else:
        console.print("[yellow]No extracted data.json found[/]")
        console.print("[dim]Run extract_parquet_data.py first for faster offline evaluation[/]")
        from datasets import load_dataset, Audio
        dataset = load_dataset("japanese-asr/whisper_transcriptions.mls", "subset_0", split="train")
        dataset = dataset.select([0, 1])
        dataset = dataset.cast_column("audio", Audio(decode=False))
        samples = [{"audio": s["audio"]["path"], "reference": s["text"], "file_name": Path(s["audio"]["path"]).name} for s in dataset][0:config.sample_limit]

    evaluator = JapaneseASREvaluator(output_dir=output_dir)
    results_df, metrics = evaluator.evaluate(samples, save_audio=True)

    save_results(results_df, metrics, output_dir, config.output_subdir)
    console.rule("[bold green]Benchmark Completed![/]")
    rprint("\n[bold]Tip: Run extract_parquet_data.py once for 10x faster repeated evaluations![/]")


if __name__ == "__main__":
    main()