import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from speech_segments_extractor import (
    console,
    extract_speech_timestamps,
    load_audio,
)

matplotlib.use("Agg")  # Non-interactive backend for PNG generation


def str2bool(value: str) -> bool:
    """Helper to parse boolean args from command line (supports true/false, yes/no, 1/0)."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract speech segments from audio using SpeechBrain VAD (CRDNN-LibriParty). "
        "Saves per-segment WAVs, probs, charts, JSONs + global summary files."
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the input audio file (any format supported by librosa).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: ./generated/main_speech_segments_extractor)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Speech activation threshold (default: 0.5)",
    )
    parser.add_argument(
        "--neg-threshold",
        "-n",
        type=float,
        default=0.25,
        help="Speech deactivation threshold (default: 0.25)",
    )
    parser.add_argument(
        "--sampling-rate",
        "-r",
        type=int,
        default=16000,
        help="Target sampling rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--min-silence-duration-sec",
        "-s",
        type=float,
        default=0.250,
        help="Minimum silence duration to close a segment (default: 0.25)",
    )
    parser.add_argument(
        "--min-speech-duration-sec",
        "-p",
        type=float,
        default=0.250,
        help="Minimum speech duration to accept a segment (default: 0.25)",
    )
    parser.add_argument(
        "--max-speech-duration-sec",
        "-m",
        type=float,
        default=None,
        help="Maximum speech segment duration before forced split (None = no limit, default: None)",
    )
    parser.add_argument(
        "--return-seconds",
        type=str2bool,
        default=True,
        help="Return timestamps in seconds (default: True)",
    )
    parser.add_argument(
        "--time-resolution",
        type=int,
        default=2,
        help="Time resolution parameter (unused in current impl, default: 2)",
    )
    parser.add_argument(
        "--with-scores",
        type=str2bool,
        default=True,
        help="Include per-segment probability scores (default: True for outputs)",
    )
    parser.add_argument(
        "--normalize-loudness",
        type=str2bool,
        default=False,
        help="Normalize loudness (currently unused, default: False)",
    )
    parser.add_argument(
        "--include-non-speech",
        "-i",
        type=str2bool,
        default=False,
        help="Include non-speech (silence) segments (default: False)",
    )
    parser.add_argument(
        "--large-chunk-size",
        "-l",
        type=int,
        default=30,
        help="Large chunk size for VAD (default: 30)",
    )
    parser.add_argument(
        "--small-chunk-size",
        "-c",
        type=int,
        default=10,
        help="Small chunk size for VAD (default: 10)",
    )
    parser.add_argument(
        "--double-check",
        type=str2bool,
        default=True,
        help="Enable double-check in VAD (default: True)",
    )
    parser.add_argument(
        "--apply-energy-vad",
        "-e",
        type=str2bool,
        default=False,
        help="Apply energy-based VAD refinement (default: False)",
    )

    args = parser.parse_args()

    # Set default output dir if not provided (matches original stub)
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).parent / "generated" / Path(__file__).stem
        )
    OUTPUT_DIR = args.output_dir.resolve()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    segments_dir = OUTPUT_DIR / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]Processing:[/bold cyan] {args.audio_file.name}")

    # Load audio once (for slicing later)
    audio_np, sr = load_audio(str(args.audio_file), sr=args.sampling_rate)
    console.print(
        f"[bold green]✅ Loaded audio:[/bold green] {len(audio_np) / sr :.2f}s @ {sr}Hz"
    )

    # Extract segments
    with console.status("[bold blue]Extracting speech segments...[/bold blue]"):
        result = extract_speech_timestamps(
            audio_np,  # pass pre-loaded np array
            threshold=args.threshold,
            neg_threshold=args.neg_threshold,
            sampling_rate=args.sampling_rate,
            min_silence_duration_sec=args.min_silence_duration_sec,
            min_speech_duration_sec=args.min_speech_duration_sec,
            max_speech_duration_sec=args.max_speech_duration_sec,
            return_seconds=args.return_seconds,
            time_resolution=args.time_resolution,
            with_scores=args.with_scores,
            normalize_loudness=args.normalize_loudness,
            include_non_speech=args.include_non_speech,
            large_chunk_size=args.large_chunk_size,
            small_chunk_size=args.small_chunk_size,
            double_check=args.double_check,
            apply_energy_VAD=args.apply_energy_vad,
        )

    if args.with_scores:
        segments, _ = result  # global probs not saved globally per spec
    else:
        segments = result

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}")

    all_segments_json = []
    total_duration_sec = len(audio_np) / sr
    total_speech_sec = 0.0
    num_speech = 0
    num_non_speech = 0

    for seg in segments:
        seg_num = seg["num"]
        seg_dir = segments_dir / f"segment_{seg_num:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Timestamps (force sec for JSON consistency)
        start_sec = float(seg["start"]) if args.return_seconds else float(seg["start"]) / sr
        end_sec = float(seg["end"]) if args.return_seconds else float(seg["end"]) / sr
        duration_sec = float(seg["duration"])

        # Slice audio for sound.wav
        start_samp = int(start_sec * sr)
        end_samp = int(end_sec * sr)
        segment_audio_np = audio_np[start_samp:end_samp]

        # 1. sound.wav
        sound_path = seg_dir / "sound.wav"
        waveform = torch.from_numpy(segment_audio_np).unsqueeze(0)
        torchaudio.save(str(sound_path), waveform, sr)
        console.print(f"  [green]Saved[/green] {sound_path.name}")

        # Compute frame-level RMS energies (exactly matching VAD 160-sample hop)
        hop_samples = 160
        energies: list[float] = []
        if len(segment_audio_np) > 0:
            num_frames = len(seg.get("segment_probs", []))
            if num_frames == 0:
                num_frames = (len(segment_audio_np) + hop_samples - 1) // hop_samples
            for i in range(num_frames):
                start_samp = i * hop_samples
                end_samp = min(start_samp + hop_samples, len(segment_audio_np))
                frame = segment_audio_np[start_samp:end_samp]
                rms_val = float(np.sqrt(np.mean(frame**2))) if len(frame) > 0 else 0.0
                energies.append(rms_val)

        # 2. energies.json (new)
        energies_path = seg_dir / "energies.json"
        with open(energies_path, "w", encoding="utf-8") as f:
            json.dump(energies, f, indent=2)
        console.print(f"  [green]Saved[/green] {energies_path.name}")

        # 3. speech_probs.json
        if seg.get("segment_probs"):
            probs_path = seg_dir / "speech_probs.json"
            with open(probs_path, "w", encoding="utf-8") as f:
                json.dump(seg["segment_probs"], f, indent=2)
            console.print(f"  [green]Saved[/green] {probs_path.name}")

        # 4. speech_probs.png (now 2-panel chart with energy)
        if seg.get("segment_probs") and energies:
            plot_path = seg_dir / "speech_probs.png"
            probs_arr = np.array(seg["segment_probs"])
            hop_sec = 160 / sr
            times = np.arange(len(probs_arr)) * hop_sec

            energies_arr = np.array(energies)

            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Top: Speech probability
            axs[0].plot(times, probs_arr, "b-", label="Speech probability")
            axs[0].axhline(y=seg["prob"], color="red", linestyle="--",
                           label=f"Avg prob = {seg['prob']:.3f}")
            axs[0].set_title(f"Segment {seg_num} | {start_sec:.2f}-{end_sec:.2f}s "
                             f"| Duration {duration_sec:.2f}s")
            axs[0].set_ylabel("Probability")
            axs[0].set_ylim(0, 1)
            axs[0].grid(True, alpha=0.3)
            axs[0].legend()

            # Bottom: RMS Energy
            rms = float(np.sqrt(np.mean(segment_audio_np**2))) if len(segment_audio_np) > 0 else 0.0
            axs[1].plot(times, energies_arr, "g-", label="RMS Energy")
            axs[1].axhline(y=rms, color="orange", linestyle="--",
                           label=f"Avg RMS = {rms:.3f}")
            axs[1].set_ylabel("RMS Energy")
            axs[1].set_xlabel("Time (s)")
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()

            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            console.print(f"  [green]Saved[/green] {plot_path.name}")

        # 5. segment.json (enhanced - unchanged except ordering)
        rms = float(np.sqrt(np.mean(segment_audio_np**2))) if len(segment_audio_np) > 0 else 0.0
        seg_dict = dict(seg)
        seg_dict.update(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
                "average_prob": float(seg["prob"]),
                "average_rms": rms,
                "max_prob": float(max(seg["segment_probs"])) if seg.get("segment_probs") else 0.0,
                "min_prob": float(min(seg["segment_probs"])) if seg.get("segment_probs") else 0.0,
            }
        )
        seg_json_path = seg_dir / "segment.json"
        with open(seg_json_path, "w", encoding="utf-8") as f:
            json.dump(seg_dict, f, indent=2)
        console.print(f"  [green]Saved[/green] {seg_json_path.name}")

        all_segments_json.append(seg_dict)  # note: energies list is NOT duplicated here (saved separately)

        if seg["type"] == "speech":
            total_speech_sec += duration_sec
            num_speech += 1
        else:
            num_non_speech += 1

        # Log summary line
        console.print(
            f"[yellow][[/yellow] [bold white]{start_sec:.2f}[/bold white] - [bold white]{end_sec:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{duration_sec:.2f}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan] "
            f"rms=[bold blue]{rms:.3f}[/bold blue] [dim]{seg['type']}[/dim]"
        )

    # Global files
    (OUTPUT_DIR / "segments.json").write_text(
        json.dumps(all_segments_json, indent=2), encoding="utf-8"
    )
    summary = {
        "audio_file": str(args.audio_file.resolve()),
        "total_duration_sec": round(total_duration_sec, 3),
        "total_speech_duration_sec": round(total_speech_sec, 3),
        "speech_percentage": round((total_speech_sec / total_duration_sec * 100) if total_duration_sec > 0 else 0, 2),
        "total_segments": len(segments),
        "speech_segments": num_speech,
        "non_speech_segments": num_non_speech,
        "sampling_rate": sr,
        "vad_settings": {
            "threshold": args.threshold,
            "neg_threshold": args.neg_threshold,
            "min_silence_duration_sec": args.min_silence_duration_sec,
            "min_speech_duration_sec": args.min_speech_duration_sec,
            "max_speech_duration_sec": args.max_speech_duration_sec,
            "include_non_speech": args.include_non_speech,
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    console.print(f"\n[bold green]✅ Processing complete![/bold green]")
    console.print(f"Output directory: [bold]{OUTPUT_DIR}[/bold]")
    console.print(f"   • Per-segment folders under [cyan]{segments_dir}[/cyan]")
    console.print(f"   • Global files: [cyan]segments.json[/cyan], [cyan]summary.json[/cyan]")
