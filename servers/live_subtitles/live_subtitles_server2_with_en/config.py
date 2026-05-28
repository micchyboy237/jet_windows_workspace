import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "generated"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
