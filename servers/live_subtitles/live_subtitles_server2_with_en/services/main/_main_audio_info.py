import argparse
import shutil
from pathlib import Path

from audio_info import display_audio_info

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_AUDIO = str(
    Path("~/.cache/files/audio/recording_3_speakers.wav").expanduser().resolve()
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract speech segments with FireRedVAD + hybrid pre-roll"
    )

    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        help="input audio file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )

    return parser.parse_args()


def main():
    args = get_args()

    audio_path = args.audio_path
    output_dir = Path(args.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    # Quick check
    display_audio_info(audio_path)

    # With waveform
    # display_audio_info(audio_path, show_waveform=True)

    # Analyze loaded array
    # info = get_audio_info(audio_array, sr=16000)
    # print(f"VAD readiness score would factor in: {info['warnings']}")


if __name__ == "__main__":
    main()
