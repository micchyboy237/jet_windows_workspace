from argparse import ArgumentParser
from pathlib import Path

DEFAULT_AUDIO = Path(r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav")

def get_args() -> dict:
    parser = ArgumentParser(
        description="Japanese → English speech translator",
        epilog="If no audio file is provided, defaults to the sample file."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",                                          # makes it optional
        type=Path,
        default=DEFAULT_AUDIO,
        help="Path to input .wav file (optional — uses default sample if omitted)",
    )
    args = parser.parse_args()

    audio_path: Path = args.audio_path.expanduser().resolve()

    return {
        "audio_path": audio_path
    }
