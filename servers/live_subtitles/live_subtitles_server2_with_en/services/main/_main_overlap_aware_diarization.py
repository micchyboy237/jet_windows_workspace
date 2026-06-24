import argparse
import shutil
from pathlib import Path
from overlap_aware_diarization import logging, run_pipeline, print_result

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AUDIO = Path(r"C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav")


def save_results(result, output_dir: Path, audio_path: str, strategy: str, condition: str):
    """
    Save diarization results to separate files in the output directory.
    
    Creates:
    - summary.txt: Overall diarization summary
    - turns.csv: All speaker turns in CSV format
    - turns.rttm: Standard RTTM format
    - overlap_regions.csv: Only overlap regions
    - uncertain_regions.csv: Only uncertain regions
    - statistics.json: Diarization statistics
    """
    import json
    import csv
    from datetime import datetime
    
    log = logging.getLogger(__name__)
    log.info(f"Saving results to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save summary text file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 68 + "\n")
        f.write("DIARIZATION SUMMARY\n")
        f.write("=" * 68 + "\n")
        f.write(f"Audio File    : {audio_path}\n")
        f.write(f"Strategy      : {strategy}\n")
        f.write(f"Condition     : {condition}\n")
        f.write(f"Num Speakers  : {result.n_speakers}\n")
        f.write(f"Total Turns   : {len(result.turns)}\n")
        f.write(f"Clean Turns   : {len(result.clean_turns())}\n")
        f.write(f"Overlap Turns : {len(result.overlap_turns())}\n")
        f.write(f"Uncertain     : {len(result.uncertain_turns())}\n")
        f.write(f"Timestamp     : {datetime.now().isoformat()}\n")
        f.write("=" * 68 + "\n")
    log.info(f"✓ Summary saved to {summary_file}")
    
    # 2. Save all turns as CSV
    turns_file = output_dir / "turns.csv"
    with open(turns_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'duration', 'speaker', 'score', 'label'])
        for turn in result.turns:
            writer.writerow([
                f"{turn.start:.3f}",
                f"{turn.end:.3f}",
                f"{turn.duration:.3f}",
                turn.speaker,
                f"{turn.score:.3f}" if turn.score > 0 else "",
                turn.label
            ])
    log.info(f"✓ All turns saved to {turns_file}")
    
    # 3. Save overlap regions separately
    overlap_file = output_dir / "overlap_regions.csv"
    overlap_turns = result.overlap_turns()
    with open(overlap_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'duration', 'speaker', 'score'])
        for turn in overlap_turns:
            writer.writerow([
                f"{turn.start:.3f}",
                f"{turn.end:.3f}",
                f"{turn.duration:.3f}",
                turn.speaker,
                f"{turn.score:.3f}" if turn.score > 0 else ""
            ])
    log.info(f"✓ Overlap regions ({len(overlap_turns)}) saved to {overlap_file}")
    
    # 4. Save uncertain regions separately
    uncertain_file = output_dir / "uncertain_regions.csv"
    uncertain_turns = result.uncertain_turns()
    with open(uncertain_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'duration', 'speaker', 'score'])
        for turn in uncertain_turns:
            writer.writerow([
                f"{turn.start:.3f}",
                f"{turn.end:.3f}",
                f"{turn.duration:.3f}",
                turn.speaker,
                f"{turn.score:.3f}" if turn.score > 0 else ""
            ])
    log.info(f"✓ Uncertain regions ({len(uncertain_turns)}) saved to {uncertain_file}")
    
    # 5. Save statistics as JSON
    stats_file = output_dir / "statistics.json"
    total_overlap = sum(t.duration for t in overlap_turns)
    total_uncertain = sum(t.duration for t in uncertain_turns)
    total_clean = sum(t.duration for t in result.clean_turns())
    
    # Calculate per-speaker statistics
    speaker_stats = {}
    for turn in result.turns:
        if turn.speaker not in speaker_stats:
            speaker_stats[turn.speaker] = {
                'total_duration': 0,
                'turn_count': 0,
                'overlap_duration': 0,
                'uncertain_duration': 0
            }
        speaker_stats[turn.speaker]['total_duration'] += turn.duration
        speaker_stats[turn.speaker]['turn_count'] += 1
        if turn.label == 'overlap':
            speaker_stats[turn.speaker]['overlap_duration'] += turn.duration
        elif turn.label == 'uncertain':
            speaker_stats[turn.speaker]['uncertain_duration'] += turn.duration
    
    stats = {
        'audio_file': str(audio_path),
        'strategy': strategy,
        'condition': condition,
        'num_speakers': result.n_speakers,
        'thresholds': result.thresholds,
        'turn_statistics': {
            'total_turns': len(result.turns),
            'clean_turns': len(result.clean_turns()),
            'overlap_turns': len(overlap_turns),
            'uncertain_turns': len(uncertain_turns),
            'total_duration': total_clean + total_overlap + total_uncertain,
            'clean_duration': total_clean,
            'overlap_duration': total_overlap,
            'uncertain_duration': total_uncertain
        },
        'speaker_statistics': speaker_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    log.info(f"✓ Statistics saved to {stats_file}")
    
    # 6. Save RTTM format
    rttm_file = output_dir / "turns.rttm"
    from overlap_aware_diarization import export_rttm
    export_rttm(result, str(rttm_file))
    log.info(f"✓ RTTM saved to {rttm_file}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Overlap-aware speaker diarization with ECAPA-TDNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python overlap_aware_diarization.py meeting.wav -s resegment -c noisy
  python overlap_aware_diarization.py call.wav -s separate -c phone -n 2
  python overlap_aware_diarization.py studio.wav -c clean -r out.rttm
  python overlap_aware_diarization.py audio.wav -t hf_xxx -s nn
  python overlap_aware_diarization.py audio.wav -o custom_output_dir
  python overlap_aware_diarization.py audio.wav -s resegment -c noisy -n 3 -d \\
      --mn 2 --mx 6 --sd 2.0 --ss 1.0 -m 0.5
        """,
    )
    
    # Positional argument
    parser.add_argument("audio", type=str, nargs="?",
                        default=DEFAULT_AUDIO,
                        help="Path to audio file (.wav / .flac / .mp3)")
    
    # Strategy & condition
    parser.add_argument("-s", "--strategy", default="resegment",
                        choices=["nn", "resegment", "separate"],
                        help="Overlap handling strategy (default: resegment)")
    parser.add_argument("-c", "--condition", default="noisy",
                        choices=["clean", "noisy", "phone", "forensic"],
                        help="Acoustic condition for threshold selection (default: noisy)")
    
    # Speaker configuration
    parser.add_argument("-n", "--speakers", type=int, default=None,
                        dest="n_speakers",
                        help="Fix number of speakers (default: auto-detect)")
    parser.add_argument("-mn", "--min-spk", type=int, default=2,
                        help="Min speakers for auto-detection (default: 2)")
    parser.add_argument("-mx", "--max-spk", type=int, default=8,
                        help="Max speakers for auto-detection (default: 8)")
    
    # Authentication
    parser.add_argument("-t", "--token", default=None,
                        help="HuggingFace token for pyannote OSD (optional)")
    
    # Output
    parser.add_argument("-o", "--output", default=str(OUTPUT_DIR),
                        help=f"Output directory for results (default: {OUTPUT_DIR})")
    parser.add_argument("-r", "--rttm", default=None,
                        help="Output RTTM file path (optional)")
    
    # Audio segmentation parameters
    parser.add_argument("-sd", "--seg-dur", type=float, default=1.5,
                        help="Sliding window duration in seconds (default: 1.5)")
    parser.add_argument("-ss", "--seg-step", type=float, default=0.75,
                        help="Sliding window hop in seconds (default: 0.75)")
    parser.add_argument("-m", "--min-turn", type=float, default=0.3,
                        help="Minimum turn duration to keep in seconds (default: 0.3)")
    
    # Debug
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Log the configuration
    import logging
    log = logging.getLogger(__name__)
    log.info(f"Arguments parsed: strategy={args.strategy}, condition={args.condition}, "
             f"speakers={args.n_speakers}, output={args.output}")
    
    return args


def main():
    args = get_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert output to Path object
    output_dir = Path(args.output)
    log = logging.getLogger(__name__)
    log.info(f"Output directory set to: {output_dir}")
    
    result = run_pipeline(
        audio_path=args.audio,
        strategy=args.strategy,
        condition=args.condition,
        n_speakers=args.n_speakers,
        min_spk=args.min_spk,
        max_spk=args.max_spk,
        hf_token=args.token,
        rttm_path=args.rttm,
        seg_dur=args.seg_dur,
        seg_step=args.seg_step,
        min_turn_dur=args.min_turn,
    )
    
    # Print results to console
    print_result(result)
    
    # Save results to separate files in output directory
    save_results(result, output_dir, args.audio, args.strategy, args.condition)

if __name__ == "__main__":
    main()
