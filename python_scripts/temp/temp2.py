from pathlib import Path
import logging
import time
from typing import List, Tuple

from rich.logging import RichHandler
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

# ========================= CONFIG =========================
console = Console()

# Setup rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=True)],
)
logger = logging.getLogger("speaker-verification")

SAVE_DIR = Path("~/.cache/pretrained_models/spkrec-ecapa-voxceleb").expanduser().resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = str(SAVE_DIR)

# Audio directory
audio_dir = Path(
    r"~\Desktop\Jet_Files\Cloned_Repos\speechbrain\tests\samples\ASR"
).expanduser().resolve()

# List of verification pairs: (file1, file2, description)
verification_pairs: List[Tuple[str, str, str]] = [
    (
        audio_dir / "spk1_snt1.wav",
        audio_dir / "spk2_snt1.wav",
        "Different Speakers",
    ),
    (
        audio_dir / "spk1_snt1.wav",
        audio_dir / "spk1_snt2.wav",
        "Same Speaker",
    ),
    # Add more pairs here easily
]

# =========================================================


def main():
    logger.info("🚀 Starting Speaker Verification with ECAPA-TDNN (VoxCeleb)")

    start_time = time.time()

    # Load the model once (this can take a few seconds the first time)
    logger.info("📥 Loading pretrained SpeakerRecognition model...")
    model_start = time.time()
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=SAVE_DIR,
        local_strategy=LocalStrategy.COPY,
    )
    logger.info(f"✅ Model loaded in {time.time() - model_start:.2f} seconds")

    logger.info(f"🔍 Running {len(verification_pairs)} verification(s)...\n")

    # Option 1: Simple loop with rich track (nice progress + live updates)
    results = []
    for pair in track(verification_pairs, description="Verifying speaker pairs..."):
        p1, p2, desc = pair
        
        # Convert to file URI to avoid SpeechBrain split_path Windows bug
        path1 = p1.resolve().as_uri()
        path2 = p2.resolve().as_uri()
   
        
        logger.debug(f"Verifying: {p1.name} vs {p2.name}")
        
        score, prediction = verification.verify_files(path1, path2)

        pred_str = "✅ SAME speaker" if prediction else "❌ DIFFERENT speakers"
        logger.info(
            f"[bold]{desc}[/bold] → Score: [cyan]{score:.4f}[/cyan] | {pred_str}"
        )
        results.append((desc, float(score), bool(prediction)))

    # Option 2: If you prefer classic tqdm (more compact)
    # for pair in tqdm(verification_pairs, desc="Verifying pairs"):
    #     ... (same logic)

    total_time = time.time() - start_time
    logger.info(f"\n🎉 All verifications completed in {total_time:.2f} seconds!")

    # Summary table (rich makes it beautiful)
    console.rule("[bold]Verification Summary[/bold]")
    for desc, score, is_same in results:
        status = "✅ SAME" if is_same else "❌ DIFFERENT"
        console.print(f"{desc:30} | Score: {score:.4f} | {status}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("💥 An error occurred during verification:")
        console.print_exception(show_locals=False)