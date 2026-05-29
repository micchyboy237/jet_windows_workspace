"""
Audio tagging using sherpa-onnx CED (Conformer-based Event Detector) models.
Available CED models:
- mini:  sherpa-onnx-ced-mini-audio-tagging-2024-04-19  (~30 MB)
- small: sherpa-onnx-ced-small-audio-tagging-2024-04-19 (~60 MB)
- base:  sherpa-onnx-ced-base-audio-tagging-2024-04-19  (~120 MB)
Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models

Key difference from Zipformer:
  Zipformer uses:  model=AudioTaggingModelConfig(zipformer=OfflineZipformerAudioTaggingModelConfig(model=...))
  CED uses:        model=AudioTaggingModelConfig(ced=str(model_file))  ← direct string, no wrapper class
"""
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict

import sherpa_onnx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from audio_tagger_base import BaseAudioTagger, BASE_DIR

console = Console()
log = logging.getLogger(__name__)

# Model registry for CED variants
CED_MODELS: Dict[str, dict] = {
    "mini": {
        "name": "sherpa-onnx-ced-mini-audio-tagging-2024-04-19",
        "size": "~30 MB",
        "description": "Mini (fastest, lightest)",
        "expected_frames": 80,
    },
    "small": {
        "name": "sherpa-onnx-ced-small-audio-tagging-2024-04-19",
        "size": "~60 MB",
        "description": "Small (balanced speed/accuracy)",
        "expected_frames": 80,
    },
    "base": {
        "name": "sherpa-onnx-ced-base-audio-tagging-2024-04-19",
        "size": "~120 MB",
        "description": "Base (most accurate)",
        "expected_frames": 80,
    },
}


class CEDAudioTagger(BaseAudioTagger):
    """
    Audio tagger using sherpa-onnx CED models.
    
    Usage:
        tagger = CEDAudioTagger(variant="base", top_k=5)
        tagger.build()
        result = tagger.tag_file("audio.wav", Path("output"))
    """
    
    BACKEND_NAME = "CED"
    DEFAULT_VARIANT = "base"
    VALID_VARIANTS = tuple(CED_MODELS.keys())
    
    def __init__(self, variant: str = "base", top_k: int = 5):
        super().__init__(variant=variant, top_k=top_k)
        # Store model info for this variant
        self._model_info = CED_MODELS[self.variant]
        self.EXPECTED_FRAMES = self._model_info["expected_frames"]
    
    def _get_model_paths(self) -> dict:
        """Return resolved paths for the selected CED model variant."""
        model_dir = BASE_DIR / self._model_info["name"]
        return {
            "model": model_dir / "model.onnx",
            "model_int8": model_dir / "model.int8.onnx",
            "labels": model_dir / "class_labels_indices.csv",
            "test_wavs_dir": model_dir / "test_wavs",
            "model_info": self._model_info,
        }
    
    def _build_sherpa_config(
        self,
        model_file: str,
        label_file: str,
        top_k: int,
    ) -> sherpa_onnx.AudioTaggingConfig:
        """
        Build CED-specific AudioTaggingConfig.
        
        CED models use a direct string path (no wrapper class),
        unlike Zipformer which uses OfflineZipformerAudioTaggingModelConfig.
        """
        return sherpa_onnx.AudioTaggingConfig(
            model=sherpa_onnx.AudioTaggingModelConfig(
                ced=model_file,
                num_threads=1,
                debug=True,
                provider="cpu",
            ),
            labels=label_file,
            top_k=top_k,
        )


def main() -> None:
    """CLI entry point for CED audio tagging."""
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Audio Tagging with CED - tag audio files using sherpa-onnx CED models",
        epilog=(
            "Examples:\n"
            "  %(prog)s audio.wav\n"
            "  %(prog)s audio.wav --variant mini -k 10\n"
            "  %(prog)s audio.wav --variant base -o ./results\n"
            "\n"
            "Available Models:\n"
            + "\n".join(
                f"  • {k:6s}: {v['name']} ({v['size']})"
                for k, v in CED_MODELS.items()
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
        "-v", "--variant",
        choices=list(CED_MODELS.keys()),
        default="base",
        dest="variant",
        help="CED model variant to use (default: base)",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of top predictions to return (default: 5)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        dest="output_dir",
        help=f"Directory for output files (default: {OUTPUT_DIR})",
    )
    
    args = parser.parse_args()
    
    # Create and build the tagger
    tagger = CEDAudioTagger(variant=args.variant, top_k=args.top_k)
    
    # Resolve audio path
    if args.audio_path is None:
        default_wav = tagger.default_test_wav
        if default_wav.is_file():
            args.audio_path = str(default_wav)
            log.info(f"No audio path given — using default test file: [cyan]{default_wav}[/cyan]")
        else:
            console.print(
                "[red]No audio file provided and default test wav not found.[/red]\n"
                f"Expected: {default_wav}\n"
                "Download test files from the GitHub releases page."
            )
            raise SystemExit(1)
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        log.info(f"Cleaning output directory: [cyan]{output_dir}[/cyan]")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display header
    model_info = CED_MODELS[args.variant]
    console.print(Panel.fit(
        f"[bold yellow]🎵 CED Audio Tagging Tool[/bold yellow]\n"
        f"[dim]Model: {args.variant} ({model_info['size']}) | Top-K: {args.top_k}[/dim]\n"
        f"[dim]Window: {model_info['expected_frames']} frames (0.8 s) with 50% overlap[/dim]",
        border_style="blue",
    ))
    
    # Build tagger with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as p:
        task = p.add_task(
            f"[cyan]Initialising CED-{args.variant} tagger…",
            total=None
        )
        tagger.build()
        p.update(task, completed=True, description="[green]✓ Tagger ready")
    
    # Run tagging
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
