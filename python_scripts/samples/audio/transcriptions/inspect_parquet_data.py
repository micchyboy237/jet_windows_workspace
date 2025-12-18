#!/usr/bin/env python3
"""
Robust & debug-friendly inspector for Hugging Face Parquet files
Fixed: pyarrow Table.take() now receives proper Array of indices
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ────────────────────────────────── Debug Logging Setup ──────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

console = Console()


def load_parquet_metadata(file_path: Path) -> dict[str, Any]:
    """Load metadata + safe preview using pyarrow (with debug output)."""
    log.debug(f"Opening Parquet file: {file_path}")
    if not file_path.exists():
        console.print(f"[red]File not found:[/red] {file_path}")
        sys.exit(1)

    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow

    row_count = metadata.num_rows
    log.debug(f"Rows in file: {row_count:,}")
    log.debug(f"Columns detected: {schema.names}")

    # ──────── Safe preview (max 8 rows) ────────
    preview_rows = min(8, row_count)
    log.debug(f"Reading preview of {preview_rows} rows")

    if row_count == 0:
        sample_df = pd.DataFrame({col: pd.Series(dtype=schema.field(col).type.to_pandas_dtype())
                                 for col in schema.names})
        log.warning("File has 0 rows → returning empty preview DataFrame")
    else:
        # Fixed: convert range → pyarrow int64 array
        indices = pa.array(range(preview_rows), type=pa.int64())
        log.debug(f"Created pyarrow indices array: {indices.to_pylist()}")
        table = parquet_file.read().take(indices)
        sample_df = table.to_pandas()
        log.debug(f"Preview table shape: {sample_df.shape}")

    # Heuristics for ReazonSpeech / common HF ASR datasets
    text_col = next((c for c in schema.names if c in {"text", "sentence", "transcription"}), None)
    audio_col = next((c for c in schema.names if c in {"audio", "bytes"}), "audio")
    path_col = next((c for c in schema.names if "path" in c.lower()), None)

    info = {
        "path": file_path,
        "row_count": row_count,
        "column_count": len(schema.names),
        "columns": schema.names,
        "dtypes": {col: str(schema.field(col).type) for col in schema.names},
        "sample": sample_df,
        "text_col": text_col,
        "audio_col": audio_col,
        "path_col": path_col,
        "file_size_gb": file_path.stat().st_size / 1e9,
    }
    log.debug("Metadata dict prepared successfully")
    return info


def display_beautiful_info(info: dict[str, Any]) -> None:
    console.rule("[bold blue]Hugging Face Parquet Dataset Inspector[/bold blue]")

    # ──────── Summary Table ────────
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim")
    table.add_column("Value")
    table.add_row("File", info["path"].name)
    table.add_row("Rows", f"[green]{info['row_count']:,}[/green]")
    table.add_row("Columns", f"[blue]{info['column_count']}[/blue]")
    table.add_row("Size", f"[yellow]{info['file_size_gb']:.3f} GB[/yellow]")
    console.print(table)

    # ──────── Columns & Preview Table ────────
    col_table = Table(title="Columns & Types", box=box.SIMPLE_HEAVY)
    col_table.add_column("Column", style="cyan")
    col_table.add_column("PyArrow Type", style="green")
    col_table.add_column("First Row Preview", style="white")

    for col in info["columns"]:
        if info["row_count"] == 0:
            preview = "<empty dataset>"
        else:
            val = info["sample"][col].iloc[0]

            # ── Safe null/empty detection (avoids pd.isna() bug) ──
            if val is None or (hasattr(val, "__len__") and len(val) == 0):
                preview = "[dim]<null>[/dim]"
            # ── Audio column (common HF format: {'bytes': b'...', 'sampling_rate': 16000}) ──
            elif col == info["audio_col"] and isinstance(val, dict):
                audio_bytes = val.get("bytes") or val.get("array") or val.get("path")
                byte_len = len(audio_bytes) if audio_bytes is not None else 0
                rate = val.get("sampling_rate") or val.get("sample_rate")
                preview = f"audio | bytes: {byte_len:,} | rate: {rate or 'unknown'}"
            # ── Path column (usually file path string) ──
            elif col == info["path_col"] and val is not None:
                text = str(val)
                preview = text[:70] + ("..." if len(text) > 70 else "")
            # ── General text fallback ──
            else:
                text = str(val)
                preview = text[:80] + ("..." if len(text) > 80 else "")

        col_table.add_row(col, info["dtypes"][col], preview)

    console.print(col_table)

    # ──────── Sample Transcriptions (if text column exists) ────────
    if info["text_col"] and info["row_count"] > 0:
        text_samples = info["sample"][[info["text_col"]]].head(8)
        # Clean up column name display
        text_samples.columns = ["Transcription"]

        panel_title = f"[bold green]Sample Transcriptions[/bold green] → {info['text_col']}"
        console.print(
            Panel(
                text_samples.to_string(index=False),
                title=panel_title,
                border_style="bright_blue",
                padding=(1, 2),
            )
        )

    # ──────── Footer Tip ────────
    console.print("\n[dim]Tip: Use pd.read_parquet(path, columns=['audio', 'transcription']) for efficient streaming[/dim]")
    console.print("[dim]     Or datasets.load_dataset('path/to/parquet', split='train', streaming=True)[/dim]")


def main(path_str: str | None = None) -> None:
    default_path = "C:/Users/druiv/.cache/huggingface/hub/datasets--japanese-asr--whisper_transcriptions.reazon_speech_all/snapshots/96995b6abe6f447be95f4d6b7daa36476b809b46/subset_1.0/train-00000-of-00025.parquet"
    path = Path(path_str or default_path).expanduser().resolve()

    console.print(f"[bold]Inspecting:[/bold] {path.name}")
    log.debug(f"Resolved path: {path}")

    info = load_parquet_metadata(path)
    display_beautiful_info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beautiful Parquet inspector (debug mode)")
    parser.add_argument("file", nargs="?", help="Path to .parquet file")
    parser.add_argument("--quiet", action="store_true", help="Suppress debug logs")
    args = parser.parse_args()

    if args.quiet:
        log.setLevel(logging.INFO)

    main(args.file)
