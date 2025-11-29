# Jet_Windows_Workspace/python_scripts/samples/audio/evaluators/run_japanese_asr_benchmarks.py
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from japanese_asr_evaluator import JapaneseASREvaluator

console = Console()
log = console.log


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


def save_results(results_df: pd.DataFrame, metrics: dict, output_dir: Path, model_name: str) -> None:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{model_name}.csv"
    results_df.to_csv(csv_path, index=False)

    table = Table(
        title=f"[bold]Whisper large-v3 Benchmark ({model_name.upper()})[/]",
        show_header=True,
        header_style="bold blue"
    )
    table.add_column("Task", style="cyan")
    table.add_column("WER", style="red")
    table.add_column("CER", style="yellow")
    table.add_column("Samples", style="green")

    asr = metrics.get("asr", {})
    s2t = metrics.get("s2t", {})

    if asr.get("samples", 0) > 0:
        wer = asr["wer"]
        color = "bold green" if wer < 0.15 else "yellow" if wer < 0.30 else "bold red"
        table.add_row("ASR (ja→ja)", f"[{color}]{wer:.2%}[/]", f"{asr['cer']:.2%}", str(asr["samples"]))

    if s2t.get("samples", 0) > 0:
        wer = s2t["wer"]
        color = "bold green" if wer < 0.05 else "green" if wer < 0.12 else "yellow"
        table.add_row("S2T (ja→en)", f"[{color}]{wer:.2%}[/]", f"{s2t['cer']:.2%}", str(s2t["samples"]))

    console.print(table)
    log(f"Full results → [link=file://{csv_path}]{csv_path}[/]")

    # Final judgment
    asr_wer = asr.get("wer", 1.0)
    s2t_wer = s2t.get("wer", 1.0)

    if asr_wer < 0.20 and s2t_wer < 0.10:
        rprint(Panel(
            "[bold white on green] EXCELLENT! Production-ready ASR + Translation [/]",
            title="Success",
            style="green on black"
        ))
    elif asr_wer < 0.35:
        rprint(Panel(
            "[bold yellow] Good baseline — ready for fine-tuning [/]",
            style="bold yellow"
        ))
    else:
        rprint(Panel(
            "[bold red] ASR needs work — check audio quality or VAD settings [/]",
            style="bold red"
        ))


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
            max_samples=config.sample_limit,
            task="both",
            reference_ja_col="transcription/ja_gpt3.5",
            reference_en_col="transcription",
        )
    else:
        console.print("[yellow]No extracted data.json found[/]")
        console.print("[dim]Run extract_parquet_data.py first for 10x faster repeated evaluations[/]")
        from datasets import load_dataset, Audio
        dataset = load_dataset("japanese-asr/whisper_transcriptions.mls", "subset_0", split="train")
        dataset = dataset.select(range(config.sample_limit))
        dataset = dataset.cast_column("audio", Audio(decode=False))
        samples = [
            {
                "audio": s["audio"]["path"],
                "reference_ja": s["text"],
                "reference_en": "",
                "file_name": Path(s["audio"]["path"]).name,
            }
            for s in dataset
        ]

    # Explicitly set task="both" to ensure both ASR and S2T run
    evaluator = JapaneseASREvaluator(output_dir=output_dir, task="both")
    results_df, metrics = evaluator.evaluate(samples, save_audio=True)

    save_results(results_df, metrics, output_dir, config.output_subdir)
    console.rule("[bold green]Benchmark Completed![/]")
    rprint("\n[bold]Tip: Run extract_parquet_data.py once for 10x faster repeated evaluations![/]")


if __name__ == "__main__":
    main()