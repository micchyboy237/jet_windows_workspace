import argparse
import json
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
DEFAULT_AUDIO = str(
    Path(
        "~/Desktop/Jet_Files/Jet_Windows_Workspace/python_scripts/samples/audio"
        "/features/generated/speech_waves/waves/"
    )
    .expanduser()
    .resolve()
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def save_visualizations(labeler, output_dir: Path, console: Console) -> dict:
    """Generate and save all speaker visualizations.
    
    Parameters
    ----------
    labeler : SegmentSpeakerLabeler
        The speaker labeler instance with processed data.
    output_dir : Path
        Directory to save visualization files.
    console : Console
        Rich console for status output.
    
    Returns
    -------
    dict
        Paths to saved visualization files.
    """
    from speaker_visualizer import SpeakerVisualizer
    from speaker_html_visualizer import SpeakerHTMLVisualizer
    
    plots_dir = output_dir / "plots"
    html_dir = output_dir / "html"
    plots_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {
        'plots': {},
        'html': {}
    }
    
    console.print("\n[bold cyan]Generating visualization plots...[/bold cyan]")
    visualizer = SpeakerVisualizer(
        save_dir=str(plots_dir),
        dpi=150,
        style='seaborn-v0_8-darkgrid'
    )
    
    with console.status("[bold green]Creating plots...[/bold green]", spinner="dots"):
        figures = visualizer.plot_all(labeler, include_3d=False)
        for plot_name, fig in figures.items():
            if fig is not None:
                saved_files['plots'][plot_name] = str(plots_dir / f"{plot_name}.png")
    
    console.print("[bold cyan]Generating HTML dashboard...[/bold cyan]")
    html_visualizer = SpeakerHTMLVisualizer(save_dir=str(html_dir))
    
    with console.status("[bold green]Creating HTML dashboard...[/bold green]", spinner="dots"):
        dashboard_html = html_visualizer.get_dashboard_html(labeler)
        dashboard_path = html_dir / "speaker_dashboard.html"
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        saved_files['html']['dashboard'] = str(dashboard_path)
        
        plots_html = html_visualizer.get_plots_only_html(
            labeler,
            plots=['pca', 'tsne', 'heatmap', 'timeline']
        )
        plots_only_path = html_dir / "speaker_plots.html"
        with open(plots_only_path, "w", encoding="utf-8") as f:
            f.write(plots_html)
        saved_files['html']['plots_only'] = str(plots_only_path)
    
    return saved_files


def save_speaker_profiles(labeler, output_dir: Path) -> Path:
    """Save detailed speaker profiles including embeddings and centroids.
    
    Parameters
    ----------
    labeler : SegmentSpeakerLabeler
        The speaker labeler instance.
    output_dir : Path
        Directory to save the profiles.
    
    Returns
    -------
    Path
        Path to the saved profiles file.
    """
    profiles = {}
    speakers_info = labeler.get_all_speakers_info()
    
    for label, info in speakers_info.items():
        if label in labeler._speakers:
            ref = labeler._speakers[label]
            profiles[label] = {
                **info,
                "embedding_count": len(ref.embeddings),
                "embeddings": [emb.tolist() for emb in ref.embeddings],
                "centroid": ref.centroid.tolist() if ref.centroid is not None else None,
                "centroid_shape": list(ref.centroid.shape) if ref.centroid is not None else None,
                "first_seen": ref.first_seen,
                "last_seen": ref.last_seen,
                "active_duration": ref.active_duration,
                "centroid_quality": ref.centroid_quality,
            }
    
    profiles_file = output_dir / "speaker_profiles.json"
    with open(profiles_file, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    return profiles_file


def save_similarity_matrix(labeler, output_dir: Path) -> tuple:
    """Save pairwise speaker similarity matrix in multiple formats.
    
    Parameters
    ----------
    labeler : SegmentSpeakerLabeler
        The speaker labeler instance.
    output_dir : Path
        Directory to save the matrix.
    
    Returns
    -------
    tuple
        Paths to JSON and CSV files.
    """
    matrix_data = labeler.get_speaker_similarity_matrix()
    
    json_file = output_dir / "similarity_matrix.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(matrix_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    csv_file = output_dir / "similarity_matrix.csv"
    labels = matrix_data.get("labels", [])
    similarities = matrix_data.get("similarities", [])
    segment_counts = matrix_data.get("segment_counts", [])
    
    if labels and similarities:
        with open(csv_file, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Speaker", "Segments"] + labels)
            for i, label in enumerate(labels):
                row = [label, segment_counts[i]] + similarities[i]
                writer.writerow(row)
    
    return json_file, csv_file


def save_segment_timeline(segment_groups: List[Dict], output_dir: Path) -> Path:
    """Save segment-level data as time-series CSV.
    
    Parameters
    ----------
    segment_groups : List[Dict]
        Processed segment groups with matches.
    output_dir : Path
        Directory to save the timeline.
    
    Returns
    -------
    Path
        Path to the CSV file.
    """
    csv_file = output_dir / "segment_timeline.csv"
    
    with open(csv_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "segment_index", "timestamp", "duration", "primary_speaker",
            "primary_confidence", "primary_match_type", "is_new_speaker",
            "alternative_speakers", "alternative_confidences", "file_path"
        ])
        
        for group in segment_groups:
            primary_match = None
            alternatives = []
            
            for match in group["matches"]:
                if match.get("is_primary"):
                    primary_match = match
                else:
                    alternatives.append(match)
            
            if primary_match:
                alt_speakers = " | ".join([m["label"] for m in alternatives[:3]])
                alt_confs = " | ".join([f"{m['confidence']:.3f}" for m in alternatives[:3]])
                
                writer.writerow([
                    group["index"],
                    f"{group.get('timestamp', 0.0):.2f}",
                    f"{group['duration']:.2f}",
                    primary_match["label"],
                    f"{primary_match['confidence']:.4f}",
                    primary_match["match_type"],
                    primary_match.get("is_new_speaker", False),
                    alt_speakers,
                    alt_confs,
                    group["file"]
                ])
    
    return csv_file


def save_speaker_activity(labeler, segment_groups: List[Dict], output_dir: Path) -> Path:
    """Save speaker activity timeline showing when each speaker appears.
    
    Parameters
    ----------
    labeler : SegmentSpeakerLabeler
        The speaker labeler instance.
    segment_groups : List[Dict]
        Processed segment groups with matches.
    output_dir : Path
        Directory to save the activity data.
    
    Returns
    -------
    Path
        Path to the CSV file.
    """
    csv_file = output_dir / "speaker_activity.csv"
    
    activity_data = []
    for group in segment_groups:
        timestamp = group.get("timestamp", 0.0)
        for match in group["matches"]:
            activity_data.append({
                "timestamp": timestamp,
                "speaker": match["label"],
                "confidence": match["confidence"],
                "is_primary": match.get("is_primary", False),
                "match_type": match["match_type"],
                "duration": group["duration"],
            })
    
    activity_data.sort(key=lambda x: x["timestamp"])
    
    with open(csv_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "speaker", "confidence", "is_primary",
            "match_type", "segment_duration"
        ])
        
        for entry in activity_data:
            writer.writerow([
                f"{entry['timestamp']:.2f}",
                entry["speaker"],
                f"{entry['confidence']:.4f}",
                entry["is_primary"],
                entry["match_type"],
                f"{entry['duration']:.2f}",
            ])
    
    return csv_file


def save_maintenance_log(maintenance_history: List[Dict], output_dir: Path) -> Path:
    """Save maintenance operations log.
    
    Parameters
    ----------
    maintenance_history : List[Dict]
        History of maintenance operations.
    output_dir : Path
        Directory to save the log.
    
    Returns
    -------
    Path
        Path to the JSON file.
    """
    log_file = output_dir / "maintenance_log.json"
    
    log_data = {
        "total_operations": len(maintenance_history),
        "operations": maintenance_history,
        "summary": {
            "total_orphans_removed": sum(
                op.get("orphans_removed", 0) for op in maintenance_history
            ),
            "total_young_merged": sum(
                len(op.get("young_merged", [])) for op in maintenance_history
            ),
            "total_mature_merged": sum(
                len(op.get("mature_merged", [])) for op in maintenance_history
            ),
            "speaker_count_progression": [
                {
                    "before": op.get("speakers_before", 0),
                    "after": op.get("speakers_after", 0),
                    "reason": op.get("reason", "unknown"),
                }
                for op in maintenance_history
            ],
        }
    }
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    return log_file


def save_configuration(args, output_dir: Path) -> Path:
    """Save the configuration parameters used for this run."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "threshold_same": args.threshold_same,
            "threshold_possible": args.threshold_possible,
            "threshold_new_speaker": args.threshold_new_speaker,
            "input_paths": args.speakers,
            "output_dir": str(args.output_dir),
            "visualization_enabled": not args.no_viz,
        },
        "sample_rate": 16000,
        "embedding_model": args.embedding_model,
        "embedding_window": "whole",
    }
    config_file = output_dir / "configuration.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_file


def save_labeling_summary(
    labeler,
    segment_groups: List[Dict],
    results: List[Dict],
    output_dir: Path
) -> Path:
    """Save comprehensive labeling run summary."""
    primary_results = [r for r in results if r["is_primary"]]
    unique_speakers = len({r["label"] for r in results if r["label"].startswith("SPEAKER_")})
    total_duration = sum(r["duration"] for r in primary_results)
    
    confidences = [r["confidence"] for r in results]
    avg_confidence = np.mean(confidences) if confidences else 0.0
    median_confidence = np.median(confidences) if confidences else 0.0
    std_confidence = np.std(confidences) if confidences else 0.0
    
    match_types = {}
    for r in results:
        mt = r["match_type"]
        match_types[mt] = match_types.get(mt, 0) + 1
    
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "input": {
            "total_audio_files": len(segment_groups),
            "total_duration_seconds": round(total_duration, 2),
        },
        "results": {
            "total_results": len(results),
            "primary_results": len(primary_results),
            "unique_speakers_identified": unique_speakers,
            "average_matches_per_segment": round(len(results) / max(len(segment_groups), 1), 1),
        },
        "confidence_statistics": {
            "mean": round(float(avg_confidence), 4),
            "median": round(float(median_confidence), 4),
            "std_dev": round(float(std_confidence), 4),
            "min": round(min(confidences), 4) if confidences else 0.0,
            "max": round(max(confidences), 4) if confidences else 0.0,
        },
        "match_type_distribution": match_types,
        "speaker_statistics": {
            "total_speakers": labeler.speaker_count,
            "segments_processed": labeler.total_segments_processed,
            "speakers_created": labeler.total_speakers_created,
        },
        "health": labeler.get_health_status(),
    }
    
    summary_file = output_dir / "labeling_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    return summary_file


def save_enhanced_analysis(
    labeler,
    segment_groups: List[Dict],
    results: List[Dict],
    output_dir: Path
) -> Path:
    """Save enhanced analysis with all available data."""
    analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0",
        },
        "segments": segment_groups,
        "results": results,
        "speakers": labeler.get_all_speakers_info(),
        "similarity_matrix": labeler.get_speaker_similarity_matrix(),
        "potential_merges": labeler.find_potential_merges(min_similarity=0.50),
        "health": labeler.get_health_status(),
    }
    
    analysis_file = output_dir / "speaker_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    return analysis_file


def create_file_tree(output_dir: Path) -> Tree:
    """Create a Rich tree visualization of output files with clickable links."""
    tree = Tree(f"[bold cyan]📁 {output_dir.name}[/bold cyan]")
    
    json_files = sorted(output_dir.glob("*.json"))
    csv_files = sorted(output_dir.glob("*.csv"))
    
    if json_files:
        json_branch = tree.add("[green]📄 JSON Files[/green]")
        for f in json_files:
            size_kb = f.stat().st_size / 1024
            file_link = make_source_link(f, f.name)
            json_branch.add(f"{file_link} [grey]({size_kb:.1f} KB)[/grey]")
    
    if csv_files:
        csv_branch = tree.add("[yellow]📊 CSV Files[/yellow]")
        for f in csv_files:
            size_kb = f.stat().st_size / 1024
            file_link = make_source_link(f, f.name)
            csv_branch.add(f"{file_link} [grey]({size_kb:.1f} KB)[/grey]")
    
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        plots = sorted(plots_dir.glob("*.png"))
        if plots:
            plots_branch = tree.add("[magenta]📈 Plot Files[/magenta]")
            latest_plots = {}
            for p in plots:
                parts = p.name.split("_", 2)
                if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) == 8:
                    base_name = parts[2]
                elif len(parts) == 2 and parts[0].isdigit() and len(parts[0]) == 8:
                    base_name = parts[1]
                else:
                    base_name = p.name
                latest_plots[base_name] = p
            for base_name, filepath in sorted(latest_plots.items()):
                file_link = make_source_link(filepath, base_name)
                size_kb = filepath.stat().st_size / 1024
                plots_branch.add(f"{file_link} [grey]({size_kb:.1f} KB)[/grey]")
    
    html_dir = output_dir / "html"
    if html_dir.exists():
        html_files = sorted(html_dir.glob("*.html"))
        if html_files:
            html_branch = tree.add("[blue]🌐 HTML Files[/blue]")
            for h in html_files:
                size_kb = h.stat().st_size / 1024
                file_link = make_source_link(h, h.name)
                html_branch.add(f"{file_link} [grey]({size_kb:.1f} KB)[/grey]")
    
    return tree


def make_source_link(path, label=None):
    """Create a terminal-compatible file link."""
    path = Path(path)
    disp = label or path.name
    return f"[link=file://{escape(str(path))}][blue]{escape(str(disp))}[/blue][/link]"


def main():
    from audio_utils import resolve_audio_paths, resolve_audio_paths_as_tensor_list
    from segment_speaker_labeler import (
        SegmentSpeakerLabeler,
        DEFAULT_THRESHOLD_SAME,
        DEFAULT_THRESHOLD_POSSIBLE,
        DEFAULT_THRESHOLD_NEW_SPEAKER,
    )
    from embedding_model_factory import (
        EmbeddingModelType,
        create_embedding_model,
        list_available_models,
    )

    console = Console()
    
    parser = argparse.ArgumentParser(
        description="Speaker embedding analysis with rich terminal output."
    )
    parser.add_argument(
        "speakers",
        nargs="*",
        default=[DEFAULT_AUDIO],
        help="Paths to speaker WAV files or directories.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "-t", "--threshold-same",
        type=float,
        default=DEFAULT_THRESHOLD_SAME,
        help=f"Similarity threshold for strong match (default: {DEFAULT_THRESHOLD_SAME})",
    )
    parser.add_argument(
        "-tp", "--threshold-possible",
        type=float,
        default=DEFAULT_THRESHOLD_POSSIBLE,
        help=f"Similarity threshold for possible match (default: {DEFAULT_THRESHOLD_POSSIBLE})",
    )
    parser.add_argument(
        "-tn", "--threshold-new-speaker",
        type=float,
        default=DEFAULT_THRESHOLD_NEW_SPEAKER,
        help=f"Similarity threshold for new speaker creation (default: {DEFAULT_THRESHOLD_NEW_SPEAKER})",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="pyannote",
        choices=[e.value for e in EmbeddingModelType],
        help="Speaker embedding model backend.",
    )
    args = parser.parse_args()
    
    sample_rate = 16000
    
    console.print(Panel.fit(
        "[bold cyan]Speaker Embedding Analysis Tool[/bold cyan]\n"
        "pyannote/embedding + Dynamic Speaker Labeling",
        title="🚀 Speaker Analysis",
        border_style="blue"
    ))
    
    console.print("\n[yellow]Scanning audio files...[/yellow]")
    audio_files = resolve_audio_paths(
        args.speakers,
        recursive=True,
        includes=["**/sound.wav"],
    )
    waveforms = resolve_audio_paths_as_tensor_list(
        audio_files,
        sr=sample_rate,
    )
    audio_data = list(zip(waveforms, audio_files))
    
    MODEL_TYPE = EmbeddingModelType(args.embedding_model)

    console.print(f"[bold]Available embedding models:[/bold]")
    for name, info in list_available_models().items():
        console.print(f"  • {name} (dim={info['embedding_dim']})")

    with console.status(
        f"[bold green]Loading embedding model '{MODEL_TYPE.value}'...[/bold green]",
        spinner="dots",
    ):
        embedding_model = create_embedding_model(MODEL_TYPE)
    
    labeler = SegmentSpeakerLabeler(
        embedding_model=embedding_model,
        threshold_same=args.threshold_same,
        threshold_possible=args.threshold_possible,
        threshold_new_speaker=args.threshold_new_speaker,
        debug=True
    )
    
    console.print(f"\n[bold]Processing {len(audio_data)} audio segments...[/bold]\n")
    
    maintenance_history = []
    segment_groups = []
    
    original_maintenance = labeler.run_smart_maintenance
    
    def maintenance_wrapper(timestamp, just_created_speaker=False):
        """Wrapper to capture maintenance operations for logging."""
        result = original_maintenance(timestamp, just_created_speaker)
        if result.get("run"):
            result["segment_index"] = labeler.total_segments_processed
            result["timestamp"] = timestamp
            maintenance_history.append(result)
        return result
    
    labeler.run_smart_maintenance = maintenance_wrapper
    
    # ═══════════════════════════════════════════════════════════
    # Process all segments — labeler handles resolution internally
    # ═══════════════════════════════════════════════════════════
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Analyzing speakers...", total=len(audio_data))
        
        for i, (waveform, filepath_str) in enumerate(audio_data):
            timestamp = i * 5.0
            
            # label_segments() returns ALL segments with auto-resolved labels
            segment_groups = labeler.label_segments(
                waveform, sample_rate, timestamp
            )
            
            progress.advance(task)
    
    labeler.run_smart_maintenance = original_maintenance

    # ═══════════════════════════════════════════════════════════
    # Enrich with file metadata for display/saving
    # ═══════════════════════════════════════════════════════════
    for i, (waveform, filepath_str) in enumerate(audio_data):
        filepath = Path(filepath_str)
        group = segment_groups[i]
        group["index"] = i + 1
        group["file"] = str(filepath)
        group["filename"] = filepath.name
        group["dir"] = str(filepath.parent)
        if waveform.dim() == 1:
            group["duration"] = waveform.shape[0] / sample_rate
        else:
            group["duration"] = waveform.shape[-1] / sample_rate
    
    # ═══════════════════════════════════════════════════════════
    # Build flat results for table display
    # ═══════════════════════════════════════════════════════════
    results = []
    for group in segment_groups:
        for j, match in enumerate(group["matches"]):
            is_primary = match.get("is_primary", j == 0)
            results.append({
                "index": group["index"],
                "file": group["file"],
                "filename": group["filename"],
                "dir": group["dir"],
                "duration": group["duration"],
                "label": match["label"],
                "confidence": match["confidence"],
                "match_type": match.get("match_type", "unknown"),
                "is_primary": is_primary,
                "is_new_speaker": match.get("is_new_speaker", False),
                "is_outlier": match.get("is_outlier", False),
                "rank": 0 if is_primary else j + 1,
            })
    
    # ═══════════════════════════════════════════════════════════
    # Display results table
    # ═══════════════════════════════════════════════════════════
    table = Table(
        title="🎤 Speaker Analysis Results",
        show_lines=True,
        expand=False,
        title_justify="left"
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Dir", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Rank", justify="center")
    table.add_column("Speaker", style="green", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Match Type", justify="center")
    table.add_column("Primary", justify="center")
    table.add_column("▶️ Play", justify="center")
    
    for group in segment_groups:
        for idx, match in enumerate(group["matches"]):
            is_first = (idx == 0)
            index_str = str(group["index"]) if is_first else ""
            dir_name = Path(group["dir"]).name
            dir_link = f"[link=file://{group['dir']}]{dir_name}[/link]" if is_first else ""
            duration_str = f"{group['duration']:.2f}s" if is_first else ""
            play_link = f"[link=file://{group['file']}]▶️ Play[/link]" if is_first else ""
            
            conf_color = "green" if match["confidence"] > 0.7 else "yellow" if match["confidence"] > 0.4 else "red"
            primary_marker = "⭐" if match.get("is_primary") else ""
            # Calculate rank from position — not from a missing key
            if match.get("is_primary"):
                rank_str = "—"
            else:
                rank_str = f"#{idx}"
            
            is_outlier = match.get("is_outlier", False)
            label_style = "bold yellow" if is_outlier else "bold"
            
            table.add_row(
                index_str,
                dir_link,
                duration_str,
                rank_str,
                f"[{label_style}]{match['label']}[/{label_style}]",
                f"[{conf_color}]{match['confidence']:.3f}[/{conf_color}]",
                match.get("match_type", "unknown"),
                primary_marker,
                play_link
            )
    
    console.print(table)
    
    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    primary_results = [r for r in results if r["is_primary"]]
    unique_speakers = len({r["label"] for r in results if r["label"].startswith("SPEAKER_")})
    total_duration = sum(r["duration"] for r in primary_results)
    
    # Outlier stats
    outlier_info = ""
    if labeler.use_outlier_buffer:
        outlier_stats = labeler.get_outlier_stats_for_display()
        active = outlier_stats.get("active_outliers", 0)
        resolved = outlier_stats.get("resolved_outliers", 0)
        promoted = outlier_stats.get("total_promotions", 0)
        if active > 0:
            outlier_info += f"\nUnresolved outliers: [bold yellow]{active}[/bold yellow]"
        if resolved > 0:
            outlier_info += f"\nOutliers resolved: [bold green]{resolved}[/bold green]"
        if promoted > 0:
            outlier_info += f"\nTotal promotions: [bold cyan]{promoted}[/bold cyan]"
    
    console.print(Panel(
        f"Total segments: [bold]{len(segment_groups)}[/bold]\n"
        f"Total results (incl. alternatives): [bold]{len(results)}[/bold]\n"
        f"Total duration: [bold]{total_duration:.1f}s[/bold]\n"
        f"Unique speakers: [bold cyan]{unique_speakers}[/bold cyan]"
        f"{outlier_info}\n"
        f"Average matches per segment: [bold]{len(results) / max(len(segment_groups), 1):.1f}[/bold]",
        title="Summary",
        border_style="green",
        padding=(1, 2)
    ))
    
    # ═══════════════════════════════════════════════════════════
    # Save output files
    # ═══════════════════════════════════════════════════════════
    output_dir = args.output_dir
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("\n[bold cyan]Saving analysis results...[/bold cyan]")
    
    config_file = save_configuration(args, output_dir)
    analysis_file = save_enhanced_analysis(labeler, segment_groups, results, output_dir)
    profiles_file = save_speaker_profiles(labeler, output_dir)
    sim_json, sim_csv = save_similarity_matrix(labeler, output_dir)
    timeline_file = save_segment_timeline(segment_groups, output_dir)
    activity_file = save_speaker_activity(labeler, segment_groups, output_dir)
    maintenance_file = save_maintenance_log(maintenance_history, output_dir)
    summary_file = save_labeling_summary(labeler, segment_groups, results, output_dir)
    
    health_status = labeler.get_health_status()
    health_file = output_dir / "health_status.json"
    with open(health_file, "w", encoding="utf-8") as f:
        json.dump(health_status, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    if not args.no_viz and labeler.known_speakers:
        viz_files = save_visualizations(labeler, output_dir, console)
    elif args.no_viz:
        console.print("\n[yellow]Visualization generation skipped (--no-viz flag)[/yellow]")
    else:
        console.print("\n[yellow]No speakers found - skipping visualization[/yellow]")
    
    console.print("\n[bold green]📂 Output Files (click to open):[/bold green]")
    file_tree = create_file_tree(output_dir)
    console.print(file_tree)
    
    console.print("\n[bold green]✅ Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
