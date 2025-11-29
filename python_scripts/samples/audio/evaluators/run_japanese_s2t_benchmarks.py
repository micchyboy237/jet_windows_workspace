from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from japanese_s2t_evaluator import JapaneseS2TEvaluator

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

console = Console()
log = console.log


@dataclass(frozen=True)
class BenchmarkConfig:
    sample_limit: int = 50
    output_subdir: str = "whisper-large-v3"


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
    csv_path = results_dir / f"{model_name}_s2t.csv"
    results_df.to_csv(csv_path, index=False)

    s2t = metrics.get("s2t", {})
    wer = s2t.get("wer", 1.0)
    cer = s2t.get("cer", 1.0)
    samples = s2t.get("samples", 0)

    table = Table(
        title=f"[bold magenta]Japanese → English S2T Benchmark ({model_name.upper()})[/]",
        show_header=True,
        header_style="bold blue"
    )
    table.add_column("Task", style="cyan")
    table.add_column("WER", style="red")
    table.add_column("CER", style="yellow")
    table.add_column("Samples", style="green")

    color = "bold green" if wer < 0.08 else "green" if wer < 0.15 else "yellow" if wer < 0.25 else "bold red"
    table.add_row(
        "S2T (ja→en)",
        f"[{color}]{wer:.2%}[/]",
        f"{cer:.2%}",
        str(samples)
    )

    console.print(table)
    log(f"Full results saved → [link=file://{csv_path}]{csv_path}[/]")

    # Final judgment based on translation quality
    if wer < 0.08:
        rprint(Panel(
            "[bold white on green] OUTSTANDING Translation Quality! Production-ready S2T [/]",
            title="Excellent",
            style="green on black"
        ))
    elif wer < 0.15:
        rprint(Panel(
            "[bold white on bright_green] Very Strong — Ready for real-world use [/]",
            title="Great",
            style="bright_green on black"
        ))
    elif wer < 0.25:
        rprint(Panel(
            "[bold yellow on black] Acceptable baseline — fine-tuning recommended [/]",
            title="Fair",
            style="bold yellow"
        ))
    else:
        rprint(Panel(
            "[bold white on red] Needs improvement — check model, audio quality, or VAD [/]",
            title="Poor",
            style="bold red"
        ))


def main(config: "BenchmarkConfig" | None = None) -> None:
    config = config or BenchmarkConfig()
    console.rule("[bold magenta]Japanese → English Speech-to-Text Translation Benchmark[/]")
    log(f"Running with [bold cyan]{config.sample_limit}[/] samples")

    output_dir = make_output_dir()

    # Prefer offline extracted data.json (much faster)
    default_json = Path(__file__).parent.parent / "generated" / "extract_parquet_data" / "data.json"
    default_json = str(default_json.resolve())
    if Path(default_json).exists():
        console.print("[bold green]Found extracted data.json → using offline mode (fast!)[/]")
        samples = JapaneseS2TEvaluator.from_extracted_json(
            default_json,
            max_samples=config.sample_limit,
            reference_en_col="transcription",  # or your actual English column
        )
    else:
        console.print("[yellow]No extracted data.json found[/]")
        console.print("[dim]Tip: Run extract_parquet_data.py once for 10–20x faster repeated runs[/]")
        from datasets import load_dataset, Audio

        dataset = load_dataset("japanese-asr/whisper_transcriptions.mls", "subset_0", split="train")
        dataset = dataset.select(range(config.sample_limit))
        dataset = dataset.cast_column("audio", Audio(decode=False))

        # Note: This dataset has Japanese text → we need English references!
        # So we'll skip live loading unless you have a proper ja→en dataset
        console.print("[red]Warning: Live dataset lacks English references → using dummy fallback[/]")
        samples = [
            {
                "audio": s["audio"]["path"],
                "reference_en": "This is a placeholder English reference.",  # Replace with real data
                "file_name": Path(s["audio"]["path"]).name,
            }
            for s in dataset
        ]

    # === PASS DEVICE & COMPUTE TYPE TO EVALUATOR ===
    evaluator = JapaneseS2TEvaluator(
        model_size="large-v3",
        output_dir=output_dir,
        save_audio=True,
    )

    console.print(f"[bold blue]Starting translation of {len(samples)} Japanese audio files to English...[/]")
    results_df, metrics = evaluator.evaluate(samples, save_audio=True)
    save_results(results_df, metrics, output_dir, config.output_subdir)

    console.rule("[bold green]S2T Benchmark Completed Successfully![/]")
    rprint("\n[bold cyan]Pro Tip:[/] Run your data extraction script once to enable lightning-fast repeated evaluations!")


if __name__ == "__main__":
    sample_limit = 1
    # Run with more samples for real benchmarking
    main(BenchmarkConfig(sample_limit=sample_limit, output_subdir=OUTPUT_DIR))