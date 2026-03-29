from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
import logging


console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def extract_video_segment(
    video_path: str | Path,
    out_file: str = "video_segment.mp4",
    *,
    start: float = 0.0,
    end: Optional[float] = None,
) -> str:
    """
    Extract a segment from a video file.

    Args:
        video_path: Path to input video.
        out_file: Output file name/path.
        start: Start time in seconds.
        end: End time in seconds (optional).

    Returns:
        Absolute path to the output file.
    """
    input_path = Path(video_path).expanduser().resolve()
    output_path = Path(out_file).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if start < 0:
        raise ValueError("start must be >= 0")

    if end is not None and end <= start:
        raise ValueError("end must be greater than start")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(input_path),
    ]

    if end is not None:
        duration = end - start
        cmd.extend(["-t", str(duration)])

    cmd.extend([
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ])

    logger.info(f"Extracting segment: start={start}, end={end}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.exception("ffmpeg execution failed")
        raise RuntimeError(f"ffmpeg failed: {e}") from e

    logger.info("Extraction complete")
    return str(output_path)


def _build_parser() -> argparse.ArgumentParser:
    import shutil

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "video_segment.mp4"

    parser = argparse.ArgumentParser(
        description="Extract a segment from a video file using ffmpeg"
    )

    # Positional
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file",
    )

    # Optional args with defaults + shorthands
    parser.add_argument(
        "-o", "--out-file",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output file path (default: video_segment.mp4)",
    )
    parser.add_argument(
        "-s", "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0.0)",
    )
    parser.add_argument(
        "-e", "--end",
        type=float,
        default=None,
        help="End time in seconds (default: till end of video)",
    )

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    try:
        result = extract_video_segment(
            video_path=args.video_path,
            out_file=args.out_file,
            start=args.start,
            end=args.end,
        )
        console.print(f"[green]Saved to:[/green] [link=file://{result}]{result}[/link]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise
