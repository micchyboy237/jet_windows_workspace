# run_audio_batch_processor.py

import asyncio
from pathlib import Path

from utils.audio_utils import resolve_audio_paths  # Adjust import if needed

from audio_batch_processor import (
    AudioBatchProcessor,
    TranslationResult,
)

import logging
from rich.logging import RichHandler  # Correct import

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

log = logging.getLogger(__name__)

async def example_callback(result: TranslationResult) -> None:
    """Callback showing the new detailed information."""
    status = "[bold green]Success[/bold green]" if result["success"] else "[bold red]Failed[/bold red]"
    
    log.info(f"{status} – {Path(result['audio_path']).name}")
    log.info(f"Japanese : {result['japanese_text']}")
    log.info(f"English  : {result['translation']}")
    log.info(
        f"Timing   : {result['start_ms']}ms → {result['end_ms']}ms "
        f"({result['duration_ms']}ms)"
    )
    log.info(
        f"Scores   : logprob={result['avg_logprob']:.3f} | "
        f"no_speech={result['no_speech_prob']:.3f} | "
        f"compression={result['compression_ratio']:.2f}"
    )

async def main() -> None:
    # Configuration
    output_dir = Path(__file__).parent / "generated" / "single_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Single audio path provided
    audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server\utterance_1c2d9158_0002_20260102_201650.wav"
    
    # Validate path exists
    if not Path(audio_path).exists():
        log.error(f"[bold red]Audio file not found:[/bold red] {audio_path}")
        return

    processor = AudioBatchProcessor(
        batch_size=1,                    # Process one at a time for this demo
        max_concurrent_transcriptions=1,
        output_dir=str(output_dir),
        language="ja",
    )

    log.info("[bold cyan]Adding single audio file to processor...[/bold cyan]")
    await processor.add_audio(audio_path, callback=example_callback)

    log.info("[bold cyan]Shutting down processor (will wait for completion)...[/bold cyan]")
    await processor.shutdown(wait=True)

    log.info("[bold green]Processing complete![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())