"""
download_parquet_data.py

Minimal script to fetch and download a sample (parquet) subset from the target Japanese ASR dataset.
"""

from datasets import load_dataset, Audio
from pathlib import Path
from typing import List
import rich
from rich.logging import RichHandler
import logging

# Configure rich logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("download_parquet_data")


def _list_existing_cache_files(cache_dir: Path) -> List[Path]:
    """Return sorted list of meaningful cache files (arrow/parquet/json) in the directory."""
    if not cache_dir.exists():
        return []
    return sorted(
        [p for p in cache_dir.rglob("*") if p.suffix in {".arrow", ".parquet", ".json", ".cache"} and p.is_file()]
    )


def download_parquet_data(
    dataset_name: str,
    config_name: str = "subset_0",
    split: str = "train",
    sample_limit: int = 20,
) -> Path:
    """
    Download and cache a dataset subset, with beautiful rich output.

    Returns
    -------
    Path
        Absolute path to the cached dataset directory.
    """
    log.info(f"[bold cyan]Loading[/] [yellow]{dataset_name}[/][magenta]{config_name}[/] split=[green]{split}[/]")

    ds = load_dataset(dataset_name, config_name, split=split)

    if sample_limit is not None:
        actual_limit = min(sample_limit, len(ds))
        ds = ds.select(range(actual_limit))
        log.info(f"[bold]Limited to {actual_limit} samples[/]")

    ds = ds.cast_column("audio", Audio(decode=False))

    # Resolve final cache directory
    cache_dir = Path(ds.cache_files[0]["filename"]).parent.resolve()

    log.info(f"[bold green]Dataset cached at:[/] {cache_dir}")

    # Show existing files in cache (nice tree-like view)
    existing_files = _list_existing_cache_files(cache_dir)
    if existing_files:
        log.info("[bold]Existing cache files:[/]")
        for f in existing_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            rich.print(f"  • {f.relative_to(cache_dir)}  [dim]({size_mb:.2f} MiB)[/]")
    else:
        log.info("[dim]No cache files found yet (will be created on first access)[/]")

    return cache_dir

if __name__ == "__main__":
    dataset_name = "japanese-asr/whisper_transcriptions.reazon_speech_all"
    download_parquet_data(dataset_name)