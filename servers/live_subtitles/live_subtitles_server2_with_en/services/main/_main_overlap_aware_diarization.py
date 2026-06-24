import argparse
import shutil
import torch
from typing import List
from pathlib import Path
from overlap_aware_diarization import DiarizationResult, logging, log, run_pipeline, print_result

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AUDIO = Path(r"C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav")


def save_audio_segments(
    result: DiarizationResult,
    waveform: torch.Tensor,
    sr: int,
    output_dir: Path,
) -> List[dict]:
    """
    Extract and save each speaker turn as individual WAV files with metadata.
    
    Creates: output_dir/segments/segment_0001/
                ├── sound.wav
                └── meta.json
    
    Returns list of segment info dicts for display.
    """
    import json
    import soundfile as sf
    
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    
    segment_info = []
    clean_turns = [t for t in result.turns if t.label == "speech"]
    
    log.info(f"Extracting {len(clean_turns)} clean segments to {segments_dir}")
    
    for idx, turn in enumerate(clean_turns, 1):
        # Create segment subdirectory
        seg_name = f"segment_{idx:04d}"
        seg_dir = segments_dir / seg_name
        seg_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract audio segment
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment_audio = waveform[:, start_sample:end_sample].squeeze().numpy()
        
        # Save WAV file
        wav_path = seg_dir / "sound.wav"
        sf.write(str(wav_path), segment_audio, sr)
        
        # Create metadata
        meta = {
            "segment_id": idx,
            "segment_name": seg_name,
            "speaker": turn.speaker,
            "start_time": round(turn.start, 3),
            "end_time": round(turn.end, 3),
            "duration": round(turn.duration, 3),
            "confidence_score": round(turn.score, 3) if turn.score > 0 else None,
            "label": turn.label,
            "audio_file": str(wav_path),
            "sample_rate": sr,
            "num_samples": len(segment_audio)
        }
        
        meta_path = seg_dir / "meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        segment_info.append({
            "segment_num": idx,
            "segment_name": seg_name,
            "speaker": turn.speaker,
            "start": turn.start,
            "end": turn.end,
            "duration": turn.duration,
            "score": turn.score,
            "label": turn.label,
            "wav_path": str(wav_path),
            "meta_path": str(meta_path),
        })
        
        log.debug(f"  ✓ {seg_name}: {turn.speaker} {turn.start:.2f}s-{turn.end:.2f}s "
                  f"({turn.duration:.2f}s, {len(segment_audio)} samples)")
    
    log.info(f"✓ Saved {len(segment_info)} segments to {segments_dir}")
    return segment_info


def save_results(result, output_dir: Path, audio_path: str, strategy: str, condition: str, 
                 waveform=None, sr=None):
    """
    Save diarization results to separate files in the output directory.
    Creates:
    - summary.txt: Overall diarization summary
    - turns.csv: All speaker turns in CSV format
    - turns.rttm: Standard RTTM format
    - overlap_regions.csv: Only overlap regions
    - uncertain_regions.csv: Only uncertain regions
    - statistics.json: Diarization statistics
    - segments/segment_XXXX/: Individual speaker segments with audio and metadata
    """
    import json
    import csv
    from datetime import datetime
    
    log.info(f"Saving results to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save segment audio files if waveform is provided
    segment_info = []
    if waveform is not None and sr is not None:
        segment_info = save_audio_segments(result, waveform, sr, output_dir)
    
    # Summary file
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
        f.write(f"Segments      : {len(segment_info)}\n")  # Added
        f.write(f"Timestamp     : {datetime.now().isoformat()}\n")
        f.write("=" * 68 + "\n")
    log.info(f"✓ Summary saved to {summary_file}")
    
    # Turns CSV with segment number
    turns_file = output_dir / "turns.csv"
    with open(turns_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_num', 'start', 'end', 'duration', 'speaker', 'score', 'label', 'audio_file'])
        for i, turn in enumerate(result.turns, 1):
            # Find matching segment info for the audio path
            audio_path_segment = ""
            if turn.label == "speech":
                for seg in segment_info:
                    if abs(seg['start'] - turn.start) < 0.01 and abs(seg['end'] - turn.end) < 0.01:
                        audio_path_segment = seg['wav_path']
                        break
            
            writer.writerow([
                i,
                f"{turn.start:.3f}",
                f"{turn.end:.3f}",
                f"{turn.duration:.3f}",
                turn.speaker,
                f"{turn.score:.3f}" if turn.score > 0 else "",
                turn.label,
                audio_path_segment
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

    return segment_info


def display_results_table(result, segment_info):
    """
    Display results table with segment numbers and play buttons.
    """
    bar = "─" * 100
    print(f"\n{bar}")
    print(f"  {'SEG#':>5}  {'START':>8}   {'END':>8}   {'DUR':>6}   {'SCORE':>6}   "
          f"{'LABEL':<12}  {'SPEAKER':<12}  {'PLAY':>6}")
    print(bar)
    
    clean_turns = [t for t in result.turns if t.label == "speech"]
    
    for i, turn in enumerate(result.turns):
        score_str = f"{turn.score:.3f}" if turn.score > 0 else "  —  "
        tag = f"[{turn.label}]" if turn.label != "speech" else ""
        
        # Determine segment number (only for clean speech)
        seg_num = ""
        play_btn = ""
        if turn.label == "speech":
            # Find matching segment
            for seg in segment_info:
                if abs(seg['start'] - turn.start) < 0.01 and abs(seg['end'] - turn.end) < 0.01:
                    seg_num = f"{seg['segment_num']:04d}"
                    play_btn = "▶️"
                    break
        
        print(f"  {seg_num:>5}  {turn.start:>7.2f}s  {turn.end:>7.2f}s  "
              f"{turn.duration:>5.2f}s  {score_str:>6}  "
              f"{tag:<12}  {turn.speaker:<12}  {play_btn:>6}")
    
    print(bar)
    
    # Summary statistics
    total_overlap = sum(t.duration for t in result.turns if t.label == "overlap")
    total_uncertain = sum(t.duration for t in result.turns if t.label == "uncertain")
    
    print(f"  Turns       : {len(result.turns)} total  |  "
          f"{len(result.clean_turns())} clean  |  "
          f"{len(result.overlap_turns())} overlap  |  "
          f"{len(result.uncertain_turns())} uncertain")
    print(f"  Segments    : {len(segment_info)} saved to segments/")
    print(f"  Overlap dur : {total_overlap:.2f}s")
    print(f"  Uncertain   : {total_uncertain:.2f}s")
    print(f"{bar}\n")


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
    
    result, waveform, sr = run_pipeline(
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
    
    # Save results and get segment info
    segment_info = save_results(result, output_dir, args.audio, args.strategy, args.condition,
                               waveform=waveform, sr=sr)
    
    # Display enhanced results table
    print_result(result, segment_info)

if __name__ == "__main__":
    main()
