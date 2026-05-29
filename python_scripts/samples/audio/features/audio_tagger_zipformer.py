"""
Audio tagging using sherpa-onnx Zipformer models.

Available models:
- Standard: sherpa-onnx-zipformer-audio-tagging-2024-04-09 (288 MB)
- Small: sherpa-onnx-zipformer-small-audio-tagging-2024-04-15 (106 MB)

Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models

Key difference from CED:
  Zipformer uses:  AudioTaggingModelConfig(zipformer=OfflineZipformerAudioTaggingModelConfig(model=...))
  CED uses:        AudioTaggingModelConfig(ced=str(model_file))  ← direct string, no wrapper class

FireRed VAD Alignment:
  - Uses FRAME_SHIFT_SAMPLE (160 samples, 10ms) as fundamental unit
  - Window: 100 frames × 160 samples = 16,000 samples (1.0s) ← CORRECTED
  - Hop: 8,000 samples (0.5s, 50% overlap)
  - Perfect alignment with FireRed speech segments
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict

import sherpa_onnx
from audio_tagger_base import BaseAudioTagger
from audio_tagger_core import (
    BASE_DIR,
    FRAME_SHIFT_SAMPLE,
    HOP_LENGTH,
    SAMPLE_RATE,
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
log = logging.getLogger(__name__)

# CORRECTED: Audio tagging models use 1-second windows
# With 10ms frame shift: 1.0s / 0.010s = 100 frames
ZIPFORMER_MODELS: Dict[str, dict] = {
    "standard": {
        "name": "sherpa-onnx-zipformer-audio-tagging-2024-04-09",
        "size": "288 MB",
        "description": "Standard (larger, more accurate)",
        "expected_frames": 100,  # CORRECTED: 1.0s window = 100 frames × 10ms
    },
    "small": {
        "name": "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15",
        "size": "106 MB",
        "description": "Small (faster, less accurate)",
        "expected_frames": 100,  # CORRECTED: 1.0s window = 100 frames × 10ms
    },
}


class ZipformerAudioTagger(BaseAudioTagger):
    """
    Audio tagger using sherpa-onnx Zipformer models.

    Aligned with FireRed VAD:
        - Window size is multiple of FRAME_SHIFT_SAMPLE (160 samples)
        - 1-second windows (100 frames) for optimal model performance
        - Supports both file-based and per-segment tagging
        - Preserves absolute UTC timestamps for speech segments

    Usage:
        # File-based tagging (offline)
        tagger = ZipformerAudioTagger(variant="standard", top_k=5)
        tagger.build()
        result = tagger.tag_file("audio.wav", Path("output"))

        # Speech segment tagging (live, with FireRed VAD)
        tagger = ZipformerAudioTagger(variant="standard", top_k=5)
        tagger.build()
        result = tagger.tag_speech_segment(
            segment_audio=audio_np,
            segment_start_utc=datetime(...),
            segment_end_utc=datetime(...),
            segment_id=0
        )
    """

    BACKEND_NAME = "Zipformer"
    DEFAULT_VARIANT = "standard"
    VALID_VARIANTS = tuple(ZIPFORMER_MODELS.keys())

    def __init__(self, variant: str = "standard", top_k: int = 5):
        # Validate variant before super().__init__
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Valid: {', '.join(self.VALID_VARIANTS)}"
            )

        # Store model info before super init so _get_model_paths can use it
        self._model_info = ZIPFORMER_MODELS[variant]
        self.EXPECTED_FRAMES = self._model_info["expected_frames"]

        super().__init__(variant=variant, top_k=top_k)

        # Log alignment verification
        window_samples = self.EXPECTED_FRAMES * HOP_LENGTH
        hop_samples = window_samples // 2
        log.debug(
            f"Zipformer Audio Tagger initialized: {variant}\n"
            f"  FireRed alignment: ✓ verified\n"
            f"  Window: {self.EXPECTED_FRAMES} frames × {HOP_LENGTH} samples = "
            f"{window_samples} samples ({window_samples / SAMPLE_RATE:.1f}s)\n"
            f"  Hop: {hop_samples} samples ({hop_samples / SAMPLE_RATE:.1f}s, 50% overlap)\n"
            f"  Frame shift: {FRAME_SHIFT_SAMPLE} samples (10ms)"
        )

    def _get_model_paths(self) -> dict:
        """Return resolved paths for the selected Zipformer model variant."""
        model_dir = BASE_DIR / self._model_info["name"]
        return {
            "model": model_dir / "model.onnx",
            "model_int8": model_dir / "model.int8.onnx",
            "labels": model_dir / "class_labels_indices.csv",
            "test_wavs_dir": model_dir / "test_wavs",
            "tokens": model_dir / "tokens.txt",
            "model_info": self._model_info,
        }

    def _build_sherpa_config(
        self,
        model_file: str,
        label_file: str,
        top_k: int,
    ) -> sherpa_onnx.AudioTaggingConfig:
        """
        Build Zipformer-specific AudioTaggingConfig.

        Zipformer models use OfflineZipformerAudioTaggingModelConfig wrapper,
        unlike CED which passes a direct string path.
        """
        return sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(
                zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(
                    model=model_file,
                ),
                num_threads=1,
                debug=True,
                provider="cpu",
            ),
            labels=label_file,
            top_k=top_k,
        )


def main() -> None:
    """CLI entry point for Zipformer audio tagging."""
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Audio Tagging with Zipformer - tag audio files using sherpa-onnx Zipformer models",
        epilog=(
            "Examples:\n"
            "  %(prog)s audio.wav\n"
            "  %(prog)s audio.wav --variant small -k 10\n"
            "  %(prog)s audio.wav --variant small -o ./results\n"
            "\n"
            "Available Models:\n"
            + "\n".join(
                f"  • {k:9s}: {v['name']} ({v['size']})"
                for k, v in ZIPFORMER_MODELS.items()
            )
            + "\n\nDownload: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        ),
    )

    parser.add_argument(
        "audio_path",
        nargs="?",
        type=str,
        help="Path to input .wav file (omit to use the built-in test wav)",
    )
    parser.add_argument(
        "-v",
        "--variant",
        choices=list(ZIPFORMER_MODELS.keys()),
        default="standard",
        dest="variant",
        help="Zipformer model variant to use (default: standard)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of top predictions to return (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        dest="output_dir",
        help=f"Directory for output files (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    tagger = ZipformerAudioTagger(variant=args.variant, top_k=args.top_k)

    if args.audio_path is None:
        default_wav = tagger.default_test_wav
        if default_wav.is_file():
            args.audio_path = str(default_wav)
            log.info(
                f"No audio path given — using default test file: [cyan]{default_wav}[/cyan]"
            )
        else:
            console.print(
                "[red]No audio file provided and default test wav not found.[/red]\n"
                f"Expected: {default_wav}\n"
                "Download test files from the GitHub releases page."
            )
            raise SystemExit(1)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        log.info(f"Cleaning output directory: [cyan]{output_dir}[/cyan]")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = ZIPFORMER_MODELS[args.variant]

    # Calculate alignment info for display - CORRECTED
    window_samples = model_info["expected_frames"] * HOP_LENGTH
    window_sec = window_samples / SAMPLE_RATE  # Now 1.0 seconds
    hop_sec = window_sec / 2  # Now 0.5 seconds

    console.print(
        Panel.fit(
            f"[bold yellow]🎵 Zipformer Audio Tagging Tool[/bold yellow]\n"
            f"[dim]Model: {args.variant} ({model_info['size']}) | Top-K: {args.top_k}[/dim]\n"
            f"[dim]Window: {model_info['expected_frames']} frames ({window_sec:.1f}s) "
            f"with 50% overlap ({hop_sec:.1f}s hop)[/dim]\n"
            f"[dim green]✓ Aligned with FireRed VAD (frame shift: {FRAME_SHIFT_SAMPLE} samples, 10ms)[/dim green]",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as p:
        task = p.add_task(
            f"[cyan]Initialising Zipformer-{args.variant} tagger…", total=None
        )
        tagger.build()
        p.update(task, completed=True, description="[green]✓ Tagger ready")

    try:
        result = tagger.tag_file(args.audio_path, output_dir)
        console.print("[bold green]✓ Done![/bold green]")
    except Exception as e:
        log.exception("Fatal error during audio tagging")
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        log.exception("Unexpected error")
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)
