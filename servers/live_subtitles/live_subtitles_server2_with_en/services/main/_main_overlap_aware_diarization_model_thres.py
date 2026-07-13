import argparse
import shutil
import torch
from typing import List
from pathlib import Path
from overlap_aware_diarization_model_thres import (
    DiarizationResult, logging, log, run_pipeline,
    EmbeddingModelType,
)
from embedding_model_factory import list_available_models

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_AUDIO = Path(r"C:\Users\druiv\.cache\files\audio\recording_3_speakers.wav")


def make_clickable_link(uri: str, label: str = None) -> str:
    """
    Create an ANSI OSC 8 hyperlink for modern terminals.
    
    Supported in: Windows Terminal, iTerm2, VS Code terminal, 
                  GNOME Terminal, Konsole, etc.
    
    Parameters
    ----------
    uri : str
        The target URI (file:///, https://, etc.)
    label : str, optional
        Display text for the link. If None, uses uri.
    
    Returns
    -------
    str
        ANSI escape sequence for clickable link
    """
    if label is None:
        label = uri
    return f"\033]8;;{uri}\033\\{label}\033]8;;\033\\"


def save_audio_segments(
    result: DiarizationResult,
    waveform: torch.Tensor,
    sr: int,
    output_dir: Path,
) -> List[dict]:
    """
    Extract and save ALL speaker turns as individual WAV files with metadata.
    
    All turns (clean and uncertain) are numbered sequentially as they appear
    in the timeline, using 3-digit segment numbers (001-999).
    
    Creates:
        output_dir/segments/segment_001/
                    ├── sound.wav
                    └── meta.json
        output_dir/outliers/segment_003/
                    ├── sound.wav
                    └── meta.json
    
    Returns list of segment info dicts for display, ordered by start time.
    """
    import json
    import soundfile as sf

    segments_dir = output_dir / "segments"
    outliers_dir = output_dir / "outliers"
    segments_dir.mkdir(parents=True, exist_ok=True)
    outliers_dir.mkdir(parents=True, exist_ok=True)

    segment_info = []
    
    # Sort all turns by start time to maintain chronological order
    all_turns_sorted = sorted(result.turns, key=lambda t: t.start)
    
    clean_count = sum(1 for t in all_turns_sorted if t.label == "speech")
    outlier_count = sum(1 for t in all_turns_sorted if t.label != "speech")
    
    log.info(f"Extracting {len(all_turns_sorted)} total segments "
             f"({clean_count} clean, {outlier_count} outlier) "
             f"to {output_dir}")
    
    for idx, turn in enumerate(all_turns_sorted, 1):
        # Sequential 3-digit numbering for ALL segments
        seg_num = idx
        seg_name = f"segment_{seg_num:03d}"
        
        # Determine target directory based on label
        if turn.label == "speech":
            seg_dir = segments_dir / seg_name
            seg_type = "clean"
        else:
            seg_dir = outliers_dir / seg_name
            seg_type = "outlier"
        
        seg_dir.mkdir(parents=True, exist_ok=True)

        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment_audio = waveform[:, start_sample:end_sample].squeeze().numpy()

        wav_path = seg_dir / "sound.wav"
        sf.write(str(wav_path), segment_audio, sr)

        meta = {
            "segment_id": seg_num,
            "segment_name": seg_name,
            "speaker": turn.speaker,
            "start_time": round(turn.start, 3),
            "end_time": round(turn.end, 3),
            "duration": round(turn.duration, 3),
            "confidence_score": round(turn.score, 3) if turn.score > 0 else None,
            "label": turn.label,
            "audio_file": str(wav_path),
            "sample_rate": sr,
            "num_samples": len(segment_audio),
            "type": seg_type,
            "global_order": idx,
        }
        
        meta_path = seg_dir / "meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        segment_info.append({
            "segment_num": seg_num,
            "segment_name": seg_name,
            "speaker": turn.speaker,
            "start": turn.start,
            "end": turn.end,
            "duration": turn.duration,
            "score": turn.score,
            "label": turn.label,
            "wav_path": str(wav_path),
            "meta_path": str(meta_path),
            "folder_path": str(seg_dir),
            "type": seg_type,
            "global_order": idx,
        })
        
        type_tag = "[outlier]" if seg_type == "outlier" else "[clean]  "
        log.debug(f"  ✓ {seg_name} {type_tag}: {turn.speaker} "
                  f"{turn.start:.2f}s-{turn.end:.2f}s "
                  f"({turn.duration:.2f}s, {len(segment_audio)} samples)")

    clean_saved = len([s for s in segment_info if s['type'] == 'clean'])
    outlier_saved = len([s for s in segment_info if s['type'] == 'outlier'])
    log.info(f"✓ Saved {clean_saved} clean segments to {segments_dir}")
    log.info(f"✓ Saved {outlier_saved} outlier segments to {outliers_dir}")
    
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
    - segments/segment_XXX/: Clean speaker segments with audio and metadata
    - outliers/segment_XXX/: Uncertain/outlier segments with audio and metadata
    
    All segments use sequential 3-digit numbering (001-999).
    """
    import json
    import csv
    from datetime import datetime

    log.info(f"Saving results to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_info = []
    if waveform is not None and sr is not None:
        segment_info = save_audio_segments(result, waveform, sr, output_dir)

    # Summary file
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 68 + "\n")
        f.write("DIARIZATION SUMMARY\n")
        f.write("=" * 68 + "\n")
        f.write(f"Audio File      : {audio_path}\n")
        f.write(f"Strategy        : {strategy}\n")
        f.write(f"Condition       : {condition}\n")
        f.write(f"Embedding Model : {result.embedding_model}\n")
        f.write(f"Num Speakers    : {result.n_speakers}\n")
        f.write(f"Total Turns     : {len(result.turns)}\n")
        f.write(f"Clean Turns     : {len(result.clean_turns())}\n")
        f.write(f"Overlap Turns   : {len(result.overlap_turns())}\n")
        f.write(f"Uncertain       : {len(result.uncertain_turns())}\n")
        f.write(f"Total Segments  : {len(segment_info)}\n")
        f.write(f"Clean Segments  : {len([s for s in segment_info if s['type'] == 'clean'])}\n")
        f.write(f"Outlier Segments: {len([s for s in segment_info if s['type'] == 'outlier'])}\n")
        f.write(f"Timestamp       : {datetime.now().isoformat()}\n")
        f.write("=" * 68 + "\n")
    log.info(f"✓ Summary saved to {summary_file}")

    # Turns CSV (includes all turns - clean, overlap, uncertain)
    turns_file = output_dir / "turns.csv"
    with open(turns_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_num', 'start', 'end', 'duration', 'speaker', 'score', 'label', 'audio_file', 'type'])
        for i, turn in enumerate(result.turns, 1):
            audio_path_segment = ""
            seg_type = ""
            seg_num = ""
            # Match turn to segment info
            for seg in segment_info:
                if abs(seg['start'] - turn.start) < 0.01 and abs(seg['end'] - turn.end) < 0.01:
                    audio_path_segment = seg['wav_path']
                    seg_type = seg['type']
                    seg_num = f"{seg['segment_num']:03d}"
                    break
            writer.writerow([
                seg_num,
                f"{turn.start:.3f}",
                f"{turn.end:.3f}",
                f"{turn.duration:.3f}",
                turn.speaker,
                f"{turn.score:.3f}" if turn.score > 0 else "",
                turn.label,
                audio_path_segment,
                seg_type
            ])
    log.info(f"✓ All turns saved to {turns_file}")

    # Overlap regions CSV
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

    # Uncertain regions CSV
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

    # Statistics JSON
    stats_file = output_dir / "statistics.json"
    total_overlap = sum(t.duration for t in overlap_turns)
    total_uncertain = sum(t.duration for t in uncertain_turns)
    total_clean = sum(t.duration for t in result.clean_turns())
    
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
        'embedding_model': result.embedding_model,
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
        'segment_statistics': {
            'total_segments': len(segment_info),
            'clean_segments': len([s for s in segment_info if s['type'] == 'clean']),
            'outlier_segments': len([s for s in segment_info if s['type'] == 'outlier']),
        },
        'segments': [
            {
                'segment_num': seg['segment_num'],
                'segment_name': seg['segment_name'],
                'type': seg['type'],
                'speaker': seg['speaker'],
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['duration'],
                'score': seg['score'],
                'label': seg['label'],
            }
            for seg in segment_info
        ],
        'timestamp': datetime.now().isoformat()
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    log.info(f"✓ Statistics saved to {stats_file}")

    # RTTM file
    rttm_file = output_dir / "turns.rttm"
    from overlap_aware_diarization import export_rttm
    export_rttm(result, str(rttm_file))
    log.info(f"✓ RTTM saved to {rttm_file}")

    return segment_info


def display_results_table(result, segment_info):
    """
    Display results table with clickable segment numbers and play buttons.
    
    Uses ANSI OSC 8 hyperlink sequences for terminal clickability.
    SEG# cells open the segment folder on ctrl+click.
    ▶️ play button plays the segment sound.wav on ctrl+click.
    
    All segments use sequential 3-digit numbering (001-999).
    Outliers are marked with * prefix.
    """
    bar = "─" * 100
    print(f"\n{bar}")
    print(f" {'SEG#':>6}  {'START':>6}   {'END':>6}   {'DUR':>6}   {'SCORE':>6}   "
          f"{'LABEL':<12}  {'SPEAKER':<8}  {'PLAY':>6}")
    print(bar)

    # Sort segment_info by global_order to maintain chronological sequence
    sorted_segments = sorted(segment_info, key=lambda s: s['global_order'])

    for seg in sorted_segments:
        score_str = f"{seg['score']:.3f}" if seg['score'] > 0 else "  —  "
        tag = f"[{seg['label']}]" if seg['label'] != "speech" else ""
        
        folder_path = seg['folder_path']
        wav_path = seg['wav_path']
        seg_num = seg['segment_num']
        
        # Create clickable segment number that opens the folder
        folder_uri = f"file:///{folder_path.replace(chr(92), '/')}"
        seg_num_display = make_clickable_link(folder_uri, f"{seg_num:03d}")
        
        # Create clickable play button that opens the wav file
        wav_uri = f"file:///{wav_path.replace(chr(92), '/')}"
        play_btn_display = make_clickable_link(wav_uri, "▶️")
        
        # Determine display prefix based on segment type
        seg_prefix = " " if seg['type'] == 'clean' else "*"
        
        print(f"  {seg_prefix}{seg_num_display}  "
              f"{seg['start']:>7.2f}s  {seg['end']:>7.2f}s  "
              f"{seg['duration']:>5.2f}s  {score_str:>6}  "
              f"{tag:<12}  {seg['speaker']:<12}  {play_btn_display:>6}")

    print(bar)
    
    total_overlap = sum(t.duration for t in result.turns if t.label == "overlap")
    total_uncertain = sum(t.duration for t in result.turns if t.label == "uncertain")
    clean_count = len([s for s in segment_info if s['type'] == 'clean'])
    outlier_count = len([s for s in segment_info if s['type'] == 'outlier'])
    
    print(f"  Turns       : {len(result.turns)} total  |  "
          f"{len(result.clean_turns())} clean  |  "
          f"{len(result.overlap_turns())} overlap  |  "
          f"{len(result.uncertain_turns())} uncertain")
    print(f"  Segments    : {clean_count} clean  |  {outlier_count} outlier  "
          f"(* = outlier)")
    print(f"  Overlap dur : {total_overlap:.2f}s")
    print(f"  Uncertain   : {total_uncertain:.2f}s")
    print(f"  💡 Ctrl+Click SEG# → open folder  |  Ctrl+Click ▶️ → play audio")
    print(f"{bar}\n")


def print_result(result: DiarizationResult, segment_info: List[dict] = None):
    """Pretty-print diarization result to stdout with clickable segment links."""
    bar = "─" * 100 if segment_info else "─" * 68
    
    print(f"\n{bar}")
    print(f"  File            : {result.audio_path}")
    print(f"  Speakers        : {result.n_speakers}")
    print(f"  Strategy        : {result.strategy}")
    print(f"  Condition       : {result.condition}")
    print(f"  Embedding Model : {result.embedding_model}")
    print(f"  Thresholds      : same>{result.thresholds.get('same', '?')}  "
          f"ambiguous>{result.thresholds.get('ambiguous_low', '?')}  "
          f"osd_gate={result.thresholds.get('osd_gate', '?')}")
    if segment_info:
        clean_count = len([s for s in segment_info if s['type'] == 'clean'])
        outlier_count = len([s for s in segment_info if s['type'] == 'outlier'])
        print(f"  Segments        : {clean_count} clean + {outlier_count} outlier saved")
    print(bar)
    
    if segment_info:
        print(f"  {'SEG#':>5}  {'START':>8}   {'END':>8}   {'DUR':>6}   {'SCORE':>6}   "
              f"{'LABEL':<12}  {'SPEAKER':<12}  {'PLAY':>6}")
        print(bar)
        
        # Sort by global_order for chronological display
        sorted_segments = sorted(segment_info, key=lambda s: s.get('global_order', s['segment_num']))
        
        for seg in sorted_segments:
            score_str = f"{seg['score']:.3f}" if seg['score'] > 0 else "  —  "
            tag = f"[{seg['label']}]" if seg['label'] != "speech" else ""
            
            folder_path = seg['folder_path']
            wav_path = seg['wav_path']
            seg_num = seg['segment_num']
            
            # Create clickable segment number that opens the folder
            folder_uri = f"file:///{folder_path.replace(chr(92), '/')}"
            seg_num_display = make_clickable_link(folder_uri, f"{seg_num:03d}")
            
            # Create clickable play button that opens the wav file
            wav_uri = f"file:///{wav_path.replace(chr(92), '/')}"
            play_btn_display = make_clickable_link(wav_uri, "▶️")
            
            # Determine display prefix based on segment type
            seg_prefix = " " if seg['type'] == 'clean' else "*"
            
            print(f"  {seg_prefix}{seg_num_display}  "
                  f"{seg['start']:>7.2f}s  {seg['end']:>7.2f}s  "
                  f"{seg['duration']:>5.2f}s  {score_str:>6}  "
                  f"{tag:<12}  {seg['speaker']:<12}  {play_btn_display:>6}")
    else:
        print(f"  {'START':>8}   {'END':>8}   {'DUR':>6}   {'SCORE':>6}   "
              f"{'LABEL':<12}  SPEAKER")
        print(bar)
        for i, t in enumerate(result.turns, 1):
            score_str = f"{t.score:.3f}" if t.score > 0 else "  —  "
            tag = f"[{t.label}]" if t.label != "speech" else ""
            print(f"  {t.start:>7.2f}s  {t.end:>7.2f}s  "
                  f"{t.duration:>5.2f}s  {score_str:>6}  "
                  f"{tag:<12}  {t.speaker}")
    
    print(bar)
    
    total_overlap = sum(t.duration for t in result.turns if t.label == "overlap")
    total_uncertain = sum(t.duration for t in result.turns if t.label == "uncertain")
    
    print(f"  Turns       : {len(result.turns)} total  |  "
          f"{len(result.clean_turns())} clean  |  "
          f"{len(result.overlap_turns())} overlap  |  "
          f"{len(result.uncertain_turns())} uncertain")
    print(f"  Overlap dur : {total_overlap:.2f}s")
    print(f"  Uncertain   : {total_uncertain:.2f}s")
    
    if segment_info:
        clean_count = len([s for s in segment_info if s['type'] == 'clean'])
        outlier_count = len([s for s in segment_info if s['type'] == 'outlier'])
        print(f"  Segments    : {clean_count} clean + {outlier_count} outlier "
              f"saved to segments/ & outliers/")
        print(f"  💡 Ctrl+Click SEG# → open folder  |  Ctrl+Click ▶️ → play audio  |  * = outlier")
    
    print(f"{bar}\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Overlap-aware speaker diarization with selectable embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python overlap_aware_diarization.py meeting.wav -s resegment -c noisy
  python overlap_aware_diarization.py call.wav -s separate -c phone -n 2
  python overlap_aware_diarization.py studio.wav -c clean -r out.rttm
  python overlap_aware_diarization.py audio.wav -t hf_xxx -s nn
  python overlap_aware_diarization.py audio.wav -o custom_output_dir
  python overlap_aware_diarization.py audio.wav --embedding-model modelscope_eres2netv2
  python overlap_aware_diarization.py audio.wav -s resegment -c noisy -n 3 -d \\
      --mn 2 --mx 6 --sd 2.0 --ss 1.0 -m 0.5
        """,
    )
    parser.add_argument("audio", type=str, nargs="?",
                        default=DEFAULT_AUDIO,
                        help="Path to audio file (.wav / .flac / .mp3)")
    parser.add_argument("-s", "--strategy", default="resegment",
                        choices=["nn", "resegment", "separate"],
                        help="Overlap handling strategy (default: resegment)")
    parser.add_argument("-c", "--condition", default="noisy",
                        choices=["clean", "noisy", "phone", "forensic"],
                        help="Acoustic condition for threshold selection (default: noisy)")
    parser.add_argument("-n", "--speakers", type=int, default=None,
                        dest="n_speakers",
                        help="Fix number of speakers (default: auto-detect)")
    parser.add_argument("-mn", "--min-spk", type=int, default=2,
                        help="Min speakers for auto-detection (default: 2)")
    parser.add_argument("-mx", "--max-spk", type=int, default=8,
                        help="Max speakers for auto-detection (default: 8)")
    parser.add_argument("-t", "--token", default=None,
                        help="HuggingFace token for pyannote OSD (optional)")
    parser.add_argument("-o", "--output", default=str(OUTPUT_DIR),
                        help=f"Output directory for results (default: {OUTPUT_DIR})")
    parser.add_argument("-r", "--rttm", default=None,
                        help="Output RTTM file path (optional)")
    parser.add_argument("-sd", "--seg-dur", type=float, default=1.5,
                        help="Sliding window duration in seconds (default: 1.5)")
    parser.add_argument("-ss", "--seg-step", type=float, default=0.75,
                        help="Sliding window hop in seconds (default: 0.75)")
    parser.add_argument("-mt", "--min-turn", type=float, default=0.3,
                        help="Minimum turn duration to keep in seconds (default: 0.3)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("-emb", "--embedding-model", type=str, default="nemo_titanet",
                        choices=[e.value for e in EmbeddingModelType],
                        dest="embedding_model",
                        help="Speaker embedding model to use (default: nemo_titanet)")
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="Torch device, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--list-models", action="store_true",
                        dest="list_models",
                        help="List available embedding models and exit")
    
    args = parser.parse_args()
    
    if args.list_models:
        models = list_available_models()
        print("\nAvailable Embedding Models:")
        print("=" * 60)
        for name, info in models.items():
            print(f"  {name:<30}  dim={info['embedding_dim']}  ({info['class']})")
        print()
        import sys
        sys.exit(0)
    
    log.info(f"Arguments parsed: strategy={args.strategy}, condition={args.condition}, "
             f"speakers={args.n_speakers}, embedding_model={args.embedding_model}, "
             f"output={args.output}")
    return args


def main():
    args = get_args()
    
    if args.debug:
        log.setLevel(logging.DEBUG)
    
    output_dir = Path(args.output)
    log.info(f"Output directory set to: {output_dir}")
    
    device = torch.device(args.device) if args.device else None
    
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
        embedding_model=args.embedding_model,
        device=device,
    )
    
    segment_info = save_results(
        result, output_dir, args.audio, args.strategy, args.condition,
        waveform=waveform, sr=sr
    )
    
    display_results_table(result, segment_info)


if __name__ == "__main__":
    main()
