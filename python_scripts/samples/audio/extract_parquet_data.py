# Jet_Windows_Workspace/python_scripts/samples/audio/evaluators/extract_parquet_data.py

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import List, Any

import pandas as pd
import pyarrow.parquet as pq

import numpy as np  # Needed for _is_na

# --- Rich logging and progress ---
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich import print as rprint

from datetime import datetime  # <--- added

log = logging.getLogger(__name__)

def _is_na(val: Any) -> bool:
    """Robust check for missing/NA values across types."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return True
    if hasattr(val, "__len__") and len(val) == 0 and not isinstance(val, str):
        return True
    try:
        return bool(pd.isna(val))
    except:
        return False

# ----------------------------------------------------------------------
# Default output directory relative to this script
# ----------------------------------------------------------------------
DEFAULT_OUTPUT_BASE = Path(__file__).parent / "generated"
SCRIPT_NAME = Path(__file__).stem
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_BASE / SCRIPT_NAME

# Clean previous runs for clean script debugging (optional/legacy)
shutil.rmtree(DEFAULT_OUTPUT_DIR, ignore_errors=True)


def _expand_parquet_inputs(inputs: List[str | Path]) -> List[Path]:
    """Robustly resolve files, symlinks, directories, and glob patterns."""
    expanded: List[Path] = []

    for item in inputs:
        orig_path = Path(item)
        try:
            path = orig_path.expanduser()

            # Case 1: Direct file (including symlinks that point to real files)
            if path.exists() and path.stat().st_size > 0:
                if path.suffix.lower() == ".parquet":
                    expanded.append(path.resolve())
                continue

            # Case 2: Directory
            if path.is_dir():
                expanded.extend(p.resolve() for p in path.rglob("*.parquet") if p.is_file())
                continue

            # Case 3: Glob pattern
            parent = path.parent if path.parent != Path(".") else Path.cwd()
            pattern = path.name
            matches = list(parent.glob(pattern))
            if not matches and "**" in pattern:
                matches = list(parent.rglob(pattern.split("**", 1)[-1]))
            for m in matches:
                if m.suffix.lower() == ".parquet" and m.exists():
                    expanded.append(m.resolve())

        except Exception as e:
            log.warning(f"Failed to process input '{item}': {e}")

    return sorted({p for p in expanded if p.exists()})


def _extract_audio_array(audio_data: dict | Any) -> tuple["np.ndarray", int]:
    import numpy as np
    import soundfile as sf
    import io

    if isinstance(audio_data, dict):
        if arr := audio_data.get("array"):
            sr = audio_data.get("sampling_rate") or audio_data.get("sample_rate") or 16000
            return np.asarray(arr), int(sr)

        if bytes_data := audio_data.get("bytes"):
            f = io.BytesIO(bytes_data)
            arr, sr = sf.read(f)
            return arr, int(sr)

        if path := audio_data.get("path"):
            p = Path(path)
            if p.exists():
                import librosa
                arr, sr = librosa.load(p, sr=None)
                return arr, int(sr)

    raise ValueError("Cannot extract audio array from column")


def extract_parquet_data(
    parquet_paths: str | Path | List[str | Path],
    output_dir: str | Path | None = None,
    n_per_group: int = 100,
    audio_col: str = "audio",
    audio_format: str = "wav",
    overwrite: bool = False,
) -> None:
    import soundfile as sf

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(parquet_paths, (str, Path)):
        parquet_paths = [parquet_paths]
    paths = _expand_parquet_inputs(parquet_paths)
    if not paths:
        log.error("No parquet files found")
        sys.exit(1)

    # --- Updated: Detect columns and build name mapping based on pandas columns ---
    table0 = pq.read_table(paths[0])
    original_text_columns = [c for c in table0.schema.names if c != audio_col]

    # Build mapping: original parquet column → pandas attribute column
    column_name_map = {}
    df_preview = pq.read_table(paths[0]).to_pandas()
    for orig in table0.schema.names:
        # Pandas column name after safe conversion
        safe = orig.replace("/", "_").replace("\\", "_")
        if safe not in df_preview.columns:
            # fallback if pandas applied additional munging
            for c in df_preview.columns:
                if c.replace("/", "_").replace("\\", "_") == safe:
                    safe = c
                    break
        column_name_map[orig] = safe

    text_columns = original_text_columns
    log.info(f"Found text columns: {text_columns}")
    log.info(f"Column mapping: {{ {', '.join(f'{k!r}: {v!r}' for k, v in column_name_map.items() if k in text_columns)} }}")

    # Load data
    needed = [audio_col] + text_columns
    df = pd.concat([pq.read_table(p, columns=needed).to_pandas() for p in paths], ignore_index=True)
    samples = df.sample(n=min(n_per_group, len(df)), random_state=42)

    # Create dirs
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # NEW — collect text column values
    json_data = {col: [] for col in text_columns}
    json_data["audio"] = []

    # --- Extraction with Rich progress bar ---
    rprint(f"[bold green]Extracting {len(samples)} samples[/bold green] from {len(paths)} parquet file(s)...\n")

    import json

    from rich.console import Console

    console = Console()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing samples...", total=len(samples))

        for idx, row in enumerate(samples.itertuples(index=False), start=1):
            stem = f"{idx:05d}"
            wav_path = audio_dir / f"{stem}.{audio_format}"

            row_values = [item for item in row]
            row_dict = dict(zip(samples.columns, row_values))
            audio_attr = column_name_map.get(audio_col, audio_col)
            audio_data = row_dict.get(audio_attr)

            if not audio_data:
                log.warning(f"Missing audio → sample {stem}")
                json_data["audio"].append(None)
                for col in text_columns:
                    text_attr = column_name_map.get(col, col)
                    value = row_dict.get(text_attr)
                    json_data[col].append(None if _is_na(value) else value)
                progress.advance(task)
                continue

            if not wav_path.exists():
                try:
                    array, sr = _extract_audio_array(audio_data)
                    sf.write(wav_path, array, samplerate=int(sr))
                    json_data["audio"].append(wav_path.name)
                except Exception as e:
                    log.error(f"Failed to write audio {stem}: {e}")
                    json_data["audio"].append(None)
                    progress.advance(task)
                    continue
            else:
                json_data["audio"].append(wav_path.name)

            # Collect text columns
            for col in text_columns:
                text_attr = column_name_map.get(col, col)
                value = row_dict.get(text_attr)
                if _is_na(value):
                    json_data[col].append(None)
                elif hasattr(value, "tolist"):
                    json_data[col].append(value.tolist())
                elif isinstance(value, (list, tuple)):
                    json_data[col].append(list(value))
                else:
                    json_data[col].append(value)

            progress.advance(task)

    # --- Save per-sample data ---
    data_path = output_dir / "data.json"
    data_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # --- Save metadata ---
    meta = {
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "source_parquet_files": [str(p) for p in paths],
        "total_samples_extracted": len(samples),
        "audio_column_original": audio_col,
        "audio_column_pandas": column_name_map.get(audio_col, audio_col),
        "text_columns_original": text_columns,
        "column_mapping": column_name_map,
        "audio_output_format": audio_format,
        "output_audio_directory": str(audio_dir.relative_to(output_dir)),
        "random_state": 42,
        "n_per_group": n_per_group,
    }

    meta_path = output_dir / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # --- Final pretty output ---
    rprint("\n[bold magenta]Extraction complete![/bold magenta]")
    rprint(f"   Saved to: [bold cyan]{output_dir.resolve()}[/bold cyan]\n")
    rprint(f"   Audio files → [green]audio/{len(list(audio_dir.iterdir()))} files[/green]")
    rprint(f"   Samples → [blue]data.json[/blue] ({len(samples)} entries)")
    rprint("   Metadata → [blue]meta.json[/blue]\n")


if __name__ == "__main__":
    import argparse

    DEFAULT_PARQUET = "/Users/jethroestrada/.cache/huggingface/hub/datasets--japanese-asr--ja_asr.reazon_speech_all/snapshots/10c81088a41b64a99a94f5847a437e248b6a963b/subset_0/train-00000-of-00261.parquet"

    parser = argparse.ArgumentParser(description="Extract audio once + one text folder per column")
    parser.add_argument(
        "parquet",
        nargs="*",
        default=[DEFAULT_PARQUET],
        help="Parquet file(s)/dir(s) – if omitted, uses the default Japanese ReazonSpeech parquet",
    )
    parser.add_argument("-o", "--output", help="Output base directory")
    parser.add_argument("-n", "--num-samples", type=int, default=100, help="How many samples to extract")
    parser.add_argument("--audio-col", default="audio")
    parser.add_argument("--format", choices=["wav", "flac"], default="wav")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # When no parquet arguments are passed, nargs="*" gives an empty list → replace with default
    parquet_paths = args.parquet or [DEFAULT_PARQUET]

    # --- Rich logging & console setup ---
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )
    log = logging.getLogger("rich")

    extract_parquet_data(
        parquet_paths=parquet_paths,
        output_dir=args.output,
        n_per_group=args.num_samples,
        audio_col=args.audio_col,
        audio_format=args.format,
        overwrite=args.overwrite,
    )
