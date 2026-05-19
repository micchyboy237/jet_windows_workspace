import argparse
import json
import shutil
from pathlib import Path

DEFAULT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_3_speakers.wav"
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def main():
    from vad_extractors import (
        extract_valley_troughs,
    )

    parser = argparse.ArgumentParser(
        description="Extract valley troughs (strong silence points) from audio or VAD probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "probs_or_audio_path",
        nargs="?",
        default=DEFAULT_AUDIO,
        type=Path,
        help="Path to audio file or .npy file containing VAD probabilities",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=Path,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "-mvd",
        "--min-valley-duration-s",
        type=float,
        default=0.25,
        help="minimum silence duration for a valley in seconds (default: 0.25)",
    )
    parser.add_argument(
        "-sw",
        "--smooth-window",
        type=int,
        default=20,
        help="smoothing window size (default: 20)",
        dest="smoothing_window",
    )
    parser.add_argument(
        "--trough-prominence",
        type=float,
        default=0.15,
        help="Min trough prominence for soft-limit splitting (default: 0.15)",
        dest="trough_prominence",
    )
    parser.add_argument(
        "--valley-threshold",
        type=float,
        default=None,
        help="Override valley threshold (None = auto)",
        dest="valley_threshold",
    )
    parser.add_argument(
        "--min-trough-offset",
        type=float,
        default=0.4,
        help="Trough must be >= this many seconds into the segment (default: 0.4)",
        dest="min_trough_offset_s",
    )
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Loading input from: {args.probs_or_audio_path}")
        troughs = extract_valley_troughs(
            probs_or_audio=args.probs_or_audio_path,
            min_valley_duration_s=args.min_valley_duration_s,
            smoothing_window=args.smoothing_window,
            trough_prominence=args.trough_prominence,
            valley_threshold=args.valley_threshold,
            min_trough_offset_s=args.min_trough_offset_s,
        )
        output_file = args.output_dir / "valley_troughs.json"

        if not troughs:
            print("No valid valley troughs found.")
        else:
            print(f"\nFound {len(troughs)} valley trough(s):\n")
            for i, trough in enumerate(troughs, 1):
                v = trough["valley"]
                print(
                    f"{i:2d}. Time: {trough['time_s']:.3f}s  "
                    f"(Global: {trough.get('global_time_s', trough['time_s']):.3f}s)"
                )
                print(
                    f"    Prob: {trough['prob']:.4f} | "
                    f"Valley Score: {v['valley_score']:.4f} | "
                    f"Trough Score: {v['trough_score']:.4f} | "
                    f"Final Score: {v['final_score']:.4f}"
                )
                print(
                    f"    Duration: {v['duration_s']:.3f}s "
                    f"({v['frame_start']}–{v['frame_end']} frames)\n"
                )
            if args.output_dir:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(troughs, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
