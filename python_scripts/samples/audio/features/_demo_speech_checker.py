"""
_demo_speech_checker.py
======================
Demonstration script for SpeechChecker that processes all test WAV files
from the Zipformer model's test_wavs directory with different configurations.

Generates:
- Individual speech check results per audio file
- Comparative analysis across all files
- Summary statistics and visualizations
- Batch processing report
"""

from __future__ import annotations
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.tree import Tree
from rich import box

from speech_checker import (
    SpeechChecker, 
    SpeechCheckResult, 
    SpeechChunk,
    SPEECH_INDICES,
    DEFAULT_SPEECH_INDICES,
)
from audio_tagger_core import BASE_DIR, log

console = Console()

# Output base directory as specified
DEMO_OUTPUT_BASE = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(DEMO_OUTPUT_BASE, ignore_errors=True)

# Test configurations to compare
TEST_CONFIGURATIONS = {
    "standard_threshold": {
        "threshold": 0.3,
        "speech_indices": DEFAULT_SPEECH_INDICES,
        "variant": "standard",
        "description": "Standard speech detection (default)"
    },
    "sensitive": {
        "threshold": 0.15,
        "speech_indices": DEFAULT_SPEECH_INDICES,
        "variant": "standard",
        "description": "Sensitive detection (low threshold)"
    },
    "strict": {
        "threshold": 0.6,
        "speech_indices": DEFAULT_SPEECH_INDICES,
        "variant": "standard",
        "description": "Strict detection (high threshold)"
    },
    "conversation_only": {
        "threshold": 0.4,
        "speech_indices": [4],  # Only conversation
        "variant": "standard",
        "description": "Conversation detection only"
    },
    "all_speech_types": {
        "threshold": 0.3,
        "speech_indices": list(range(8)),  # All speech-related indices
        "variant": "standard",
        "description": "All speech types (including babbling, synthesizer)"
    },
}


class DemoRunner:
    """
    Runs SpeechChecker demonstrations on test WAV files.
    Processes all files with multiple configurations and generates reports.
    """
    
    def __init__(self):
        self.test_wavs_dir = self._find_test_wavs()
        self.output_base = DEMO_OUTPUT_BASE
        self.results: Dict[str, Dict[str, SpeechCheckResult]] = {}
        self.start_time = datetime.now()
        
    # def _find_test_wavs(self) -> Path:
    #     """Find the test_wavs directory from the Zipformer model."""
    #     # Try standard model first
    #     test_wavs_dir = BASE_DIR / "sherpa-onnx-zipformer-audio-tagging-2024-04-09" / "test_wavs"
        
    #     for test_dir in [test_wavs_dir]:
    #         if test_dir.exists() and any(test_dir.glob("*.wav")):
    #             console.print(f"[green]✓ Found test WAVs in:[/green] {test_dir}")
    #             return test_dir
        
    #     raise FileNotFoundError(
    #         "No test_wavs directory found. Please download the Zipformer model from:\n"
    #         "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
    #     )

    def _find_test_wavs(self) -> Path:
        """Find the test_wavs directory from the Zipformer model."""
        test_wavs_dir = Path(r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\generated\last_20_segments")
        
        for test_dir in [test_wavs_dir]:
            if test_dir.exists() and any(test_dir.rglob("sound.wav")):
                console.print(f"[green]✓ Found test WAVs in:[/green] {test_dir}")
                return test_dir
        
        raise FileNotFoundError(
            "No test_wavs directory found. Please download the Zipformer model from:\n"
            "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
        )
    
    def get_test_files(self) -> List[Path]:
        """Get all WAV files from the test directory recursively."""
        wav_files = sorted(self.test_wavs_dir.rglob("*.wav"))
        console.print(f"[cyan]Found {len(wav_files)} test WAV files[/cyan]")
        return wav_files
    
    def run_all_configurations(
        self, 
        test_files: Optional[List[Path]] = None
    ) -> None:
        """
        Run all test configurations on all test files.
        
        Args:
            test_files: List of WAV files to process (uses all if None)
        """
        if test_files is None:
            test_files = self.get_test_files()
        
        if not test_files:
            console.print("[yellow]No test files to process[/yellow]")
            return
        
        total_operations = len(TEST_CONFIGURATIONS) * len(test_files)
        
        console.print(Panel.fit(
            f"[bold cyan]🚀 SpeechChecker Demo[/bold cyan]\n"
            f"[dim]Test files: {len(test_files)} | "
            f"Configurations: {len(TEST_CONFIGURATIONS)} | "
            f"Total operations: {total_operations}[/dim]",
            border_style="blue"
        ))
        
        overall_start = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            
            overall_task = progress.add_task(
                "[cyan]Overall Progress", 
                total=total_operations
            )
            
            for config_name, config in TEST_CONFIGURATIONS.items():
                progress.update(
                    overall_task,
                    description=f"[cyan]Configuration: {config_name}"
                )
                
                # Initialize checker for this configuration
                checker = SpeechChecker(
                    threshold=config["threshold"],
                    speech_indices=config["speech_indices"],
                    variant=config["variant"],
                )
                checker.build()
                
                config_results = {}
                
                for wav_file in test_files:
                    progress.update(
                        overall_task,
                        description=f"[cyan]{config_name}: {wav_file.name}"
                    )
                    
                    try:
                        # Create output directory for this file/config combination
                        output_dir = (
                            self.output_base / 
                            config_name / 
                            wav_file.stem
                        )
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Run speech check
                        result = checker.check_speech(
                            str(wav_file),
                            output_dir=output_dir,
                            save_visualizations=True,
                        )
                        
                        config_results[wav_file.stem] = result
                        
                    except Exception as e:
                        log.error(f"Error processing {wav_file.name}: {e}")
                        console.print(f"[red]✗ Failed: {wav_file.name} - {e}[/red]")
                    
                    progress.advance(overall_task)
                
                self.results[config_name] = config_results
        
        overall_elapsed = time.time() - overall_start
        
        console.print(f"\n[green]✓ All configurations completed in {overall_elapsed:.1f}s[/green]")
    
    def generate_reports(self) -> None:
        """Generate comprehensive reports from all results."""
        console.print("\n[bold cyan]📊 Generating Reports...[/bold cyan]")
        
        # 1. Individual configuration reports
        for config_name, config_results in self.results.items():
            self._generate_configuration_report(config_name, config_results)
        
        # 2. Comparative analysis
        self._generate_comparative_analysis()
        
        # 3. Summary report
        self._generate_summary_report()
        
        # 4. Overall visualizations
        self._generate_overall_visualizations()
    
    def _generate_configuration_report(
        self, 
        config_name: str, 
        results: Dict[str, SpeechCheckResult]
    ) -> None:
        """Generate report for a single configuration."""
        config = TEST_CONFIGURATIONS[config_name]
        output_dir = self.output_base / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create configuration summary
        summary = {
            "configuration": {
                "name": config_name,
                "description": config["description"],
                "threshold": config["threshold"],
                "speech_indices": config["speech_indices"],
                "speech_types": [SPEECH_INDICES[i] for i in config["speech_indices"]],
                "variant": config["variant"],
            },
            "processing_info": {
                "timestamp": self.start_time.isoformat(),
                "total_files": len(results),
                "test_wavs_directory": str(self.test_wavs_dir),
            },
            "results_summary": {
                "files_with_speech": sum(1 for r in results.values() if r.has_speech),
                "files_without_speech": sum(1 for r in results.values() if not r.has_speech),
                "average_speech_percentage": float(np.mean([
                    r.speech_percentage for r in results.values()
                ])) if results else 0.0,
                "total_speech_duration": float(sum(
                    r.speech_duration for r in results.values()
                )),
                "average_confidence": float(np.mean([
                    r.total_speech_probability for r in results.values()
                ])) if results else 0.0,
            },
            "file_results": {},
        }
        
        # Add individual file results
        for filename, result in results.items():
            summary["file_results"][filename] = {
                "has_speech": result.has_speech,
                "confidence": result.confidence_level,
                "speech_probability": result.total_speech_probability,
                "speech_duration": result.speech_duration,
                "total_duration": result.total_duration,
                "speech_percentage": result.speech_percentage,
                "num_speech_segments": len(result.speech_chunks),
                "speech_types_detected": result.speech_types_detected,
                "processing_time": result.processing_time,
            }
        
        # Save summary JSON
        summary_path = output_dir / f"{config_name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print configuration report table
        self._print_configuration_table(config_name, config, results)
        
        console.print(f"[green]✓ Saved: {summary_path}[/green]")
    
    def _print_configuration_table(
        self,
        config_name: str,
        config: dict,
        results: Dict[str, SpeechCheckResult]
    ) -> None:
        """Print formatted table for a configuration's results."""
        
        table = Table(
            title=f"📋 Results: {config['description']}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        
        table.add_column("Audio File", style="cyan", width=20)
        table.add_column("Speech", style="green", width=8)
        table.add_column("Confidence", style="yellow", width=12)
        table.add_column("Prob", style="magenta", width=8)
        table.add_column("Duration", style="blue", width=10)
        table.add_column("Segments", style="white", width=10)
        table.add_column("Types", style="dim", width=20)
        
        for filename, result in sorted(results.items()):
            # Determine speech status icon and color
            if result.has_speech:
                speech_status = "[green]✓ Yes[/green]"
                conf_color = {
                    "High": "green",
                    "Medium": "yellow",
                    "Low": "red",
                    "Very Low": "red"
                }.get(result.confidence_level, "white")
            else:
                speech_status = "[red]✗ No[/red]"
                conf_color = "red"
            
            confidence = f"[{conf_color}]{result.confidence_level}[/{conf_color}]"
            
            # Truncate long filenames
            display_name = filename[:17] + "..." if len(filename) > 20 else filename
            
            # Speech types (truncated)
            speech_types = ", ".join(result.speech_types_detected.keys())
            if len(speech_types) > 30:
                speech_types = speech_types[:27] + "..."
            
            table.add_row(
                display_name,
                speech_status,
                confidence,
                f"{result.total_speech_probability:.1%}",
                f"{result.speech_duration:.1f}s",
                str(len(result.speech_chunks)),
                speech_types if speech_types else "N/A",
            )
        
        console.print(table)
    
    def _generate_comparative_analysis(self) -> None:
        """Generate comparative analysis across all configurations."""
        output_dir = self.output_base / "comparative_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            console.print("[yellow]No results to compare[/yellow]")
            return
        
        # Get all unique filenames
        all_files = set()
        for config_results in self.results.values():
            all_files.update(config_results.keys())
        all_files = sorted(all_files)
        
        # Create comparison data
        comparison = {
            "timestamp": self.start_time.isoformat(),
            "configurations": list(TEST_CONFIGURATIONS.keys()),
            "total_files": len(all_files),
            "comparison_matrix": {},
        }
        
        # Build comparison matrix
        for filename in all_files:
            file_comparison = {}
            
            for config_name in TEST_CONFIGURATIONS.keys():
                if config_name in self.results and filename in self.results[config_name]:
                    result = self.results[config_name][filename]
                    file_comparison[config_name] = {
                        "has_speech": result.has_speech,
                        "probability": result.total_speech_probability,
                        "segments": len(result.speech_chunks),
                        "duration": result.speech_duration,
                    }
                else:
                    file_comparison[config_name] = {
                        "has_speech": False,
                        "probability": 0.0,
                        "segments": 0,
                        "duration": 0.0,
                    }
            
            comparison["comparison_matrix"][filename] = file_comparison
        
        # Save comparison JSON
        comparison_path = output_dir / "comparative_analysis.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        # Generate comparison visualizations
        self._plot_comparison_charts(all_files, comparison, output_dir)
        
        console.print(f"[green]✓ Saved: {comparison_path}[/green]")
    
    def _plot_comparison_charts(
        self,
        all_files: List[str],
        comparison: dict,
        output_dir: Path,
    ) -> None:
        """Generate comparison charts across configurations."""
        
        # Prepare data
        configs = list(TEST_CONFIGURATIONS.keys())
        n_configs = len(configs)
        n_files = len(all_files)
        
        if n_files == 0:
            return
        
        # 1. Speech Probability Heatmap
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Heatmap data
        heatmap_data = np.zeros((n_configs, n_files))
        for i, config_name in enumerate(configs):
            for j, filename in enumerate(all_files):
                heatmap_data[i, j] = comparison["comparison_matrix"][filename][config_name]["probability"]
        
        # Plot heatmap
        ax1 = axes[0, 0]
        im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(n_files))
        ax1.set_xticklabels([f[:10] for f in all_files], rotation=45, ha='right', fontsize=8)
        ax1.set_yticks(range(n_configs))
        ax1.set_yticklabels([c.replace('_', ' ').title() for c in configs], fontsize=9)
        ax1.set_title('Speech Probability by Configuration and File', fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Probability')
        
        # Add text annotations
        for i in range(n_configs):
            for j in range(n_files):
                text = ax1.text(j, i, f'{heatmap_data[i, j]:.2f}',
                              ha="center", va="center", 
                              color="white" if heatmap_data[i, j] < 0.5 else "black",
                              fontsize=7)
        
        # 2. Speech Detection Count Bar Chart
        ax2 = axes[0, 1]
        detection_counts = []
        for config_name in configs:
            count = sum(
                1 for filename in all_files 
                if comparison["comparison_matrix"][filename][config_name]["has_speech"]
            )
            detection_counts.append(count)
        
        bars = ax2.bar(range(n_configs), detection_counts, color=plt.cm.Set3(np.linspace(0, 1, n_configs)))
        ax2.set_xticks(range(n_configs))
        ax2.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45, ha='right')
        ax2.set_ylabel('Number of Files with Speech')
        ax2.set_title('Speech Detection Count by Configuration', fontweight='bold')
        ax2.set_ylim(0, n_files + 1)
        
        # Add count labels
        for bar, count in zip(bars, detection_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 3. Average Speech Duration
        ax3 = axes[1, 0]
        avg_durations = []
        for config_name in configs:
            durations = [
                comparison["comparison_matrix"][filename][config_name]["duration"]
                for filename in all_files
            ]
            avg_durations.append(np.mean(durations) if durations else 0)
        
        ax3.bar(range(n_configs), avg_durations, color=plt.cm.Pastel1(np.linspace(0, 1, n_configs)))
        ax3.set_xticks(range(n_configs))
        ax3.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45, ha='right')
        ax3.set_ylabel('Average Duration (seconds)')
        ax3.set_title('Average Speech Duration by Configuration', fontweight='bold')
        
        # 4. Speech Segments Distribution
        ax4 = axes[1, 1]
        segment_data = []
        for config_name in configs:
            segments = [
                comparison["comparison_matrix"][filename][config_name]["segments"]
                for filename in all_files
            ]
            segment_data.append(segments)
        
        bp = ax4.boxplot(segment_data, labels=[c.replace('_', ' ').title() for c in configs])
        ax4.set_ylabel('Number of Speech Segments')
        ax4.set_title('Speech Segments Distribution by Configuration', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        chart_path = output_dir / "comparison_charts.png"
        fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        console.print(f"[green]✓ Saved comparison charts: {chart_path}[/green]")
    
    def _generate_summary_report(self) -> None:
        """Generate overall summary report."""
        output_dir = self.output_base / "summary"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate overall statistics
        all_has_speech = []
        all_probabilities = []
        all_durations = []
        
        for config_results in self.results.values():
            for result in config_results.values():
                all_has_speech.append(result.has_speech)
                all_probabilities.append(result.total_speech_probability)
                all_durations.append(result.speech_duration)
        
        summary = {
            "demo_info": {
                "timestamp": self.start_time.isoformat(),
                "test_wavs_directory": str(self.test_wavs_dir),
                "configurations_tested": list(TEST_CONFIGURATIONS.keys()),
                "total_test_files": len(self.get_test_files()),
            },
            "overall_statistics": {
                "total_processing_runs": len(all_has_speech),
                "speech_detection_rate": float(np.mean(all_has_speech)) if all_has_speech else 0.0,
                "average_speech_probability": float(np.mean(all_probabilities)) if all_probabilities else 0.0,
                "median_speech_probability": float(np.median(all_probabilities)) if all_probabilities else 0.0,
                "std_speech_probability": float(np.std(all_probabilities)) if all_probabilities else 0.0,
                "total_speech_duration": float(sum(all_durations)),
                "average_speech_duration": float(np.mean(all_durations)) if all_durations else 0.0,
            },
            "configuration_comparison": {},
            "best_configuration": self._determine_best_configuration(),
        }
        
        # Add per-configuration statistics
        for config_name, config_results in self.results.items():
            config_probs = [r.total_speech_probability for r in config_results.values()]
            config_has_speech = [r.has_speech for r in config_results.values()]
            
            summary["configuration_comparison"][config_name] = {
                "description": TEST_CONFIGURATIONS[config_name]["description"],
                "threshold": TEST_CONFIGURATIONS[config_name]["threshold"],
                "detection_rate": float(np.mean(config_has_speech)) if config_has_speech else 0.0,
                "avg_probability": float(np.mean(config_probs)) if config_probs else 0.0,
                "median_probability": float(np.median(config_probs)) if config_probs else 0.0,
                "files_processed": len(config_results),
            }
        
        # Save summary
        summary_path = output_dir / "demo_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary table
        self._print_summary_table(summary)
        
        console.print(f"[green]✓ Saved: {summary_path}[/green]")
    
    def _determine_best_configuration(self) -> Dict:
        """Determine which configuration performed best based on consistency."""
        best_config = {
            "name": "standard_threshold",
            "reason": "Default configuration",
            "metrics": {}
        }
        
        best_score = -1
        
        for config_name, config_results in self.results.items():
            if not config_results:
                continue
            
            # Score based on detection consistency (moderate detection rates are better)
            detection_rate = np.mean([r.has_speech for r in config_results.values()])
            avg_prob = np.mean([r.total_speech_probability for r in config_results.values()])
            
            # Prefer configurations with moderate detection rates (0.3-0.7)
            consistency_score = 1.0 - abs(0.5 - detection_rate)
            confidence_score = avg_prob
            total_score = consistency_score * 0.6 + confidence_score * 0.4
            
            best_config["metrics"][config_name] = {
                "detection_rate": detection_rate,
                "avg_probability": avg_prob,
                "score": total_score,
            }
            
            if total_score > best_score:
                best_score = total_score
                best_config["name"] = config_name
                best_config["reason"] = (
                    f"Best balance of detection rate ({detection_rate:.1%}) "
                    f"and confidence ({avg_prob:.1%})"
                )
        
        return best_config
    
    def _print_summary_table(self, summary: Dict) -> None:
        """Print overall summary table."""
        
        console.print("\n[bold cyan]📊 DEMO SUMMARY REPORT[/bold cyan]")
        
        # Main stats table
        stats = summary["overall_statistics"]
        table = Table(title="Overall Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Processing Runs", str(stats["total_processing_runs"]))
        table.add_row("Speech Detection Rate", f"{stats['speech_detection_rate']:.1%}")
        table.add_row("Average Probability", f"{stats['average_speech_probability']:.1%}")
        table.add_row("Median Probability", f"{stats['median_speech_probability']:.1%}")
        table.add_row("Std Deviation", f"{stats['std_speech_probability']:.3f}")
        table.add_row("Total Speech Duration", f"{stats['total_speech_duration']:.1f}s")
        table.add_row("Average Duration/File", f"{stats['average_speech_duration']:.1f}s")
        
        console.print(table)
        
        # Best configuration
        best = summary["best_configuration"]
        console.print(Panel.fit(
            f"[bold green]🏆 Best Configuration:[/bold green] {best['name']}\n"
            f"[dim]{best['reason']}[/dim]",
            border_style="green"
        ))
    
    def _generate_overall_visualizations(self) -> None:
        """Generate overall visualizations for the entire demo."""
        output_dir = self.output_base / "overall_visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            return
        
        # 1. Overall speech probability distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Collect all probabilities by configuration
        config_probs = {}
        for config_name, config_results in self.results.items():
            config_probs[config_name] = [
                r.total_speech_probability for r in config_results.values()
            ]
        
        # Plot 1: Probability distribution violin plot
        ax1 = axes[0, 0]
        if config_probs:
            data = list(config_probs.values())
            labels = [c.replace('_', ' ').title() for c in config_probs.keys()]
            vp = ax1.violinplot(data, showmeans=True, showmedians=True)
            ax1.set_xticks(range(1, len(labels) + 1))
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_ylabel('Speech Probability')
            ax1.set_title('Speech Probability Distribution by Configuration', fontweight='bold')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Detection success rate pie chart
        ax2 = axes[0, 1]
        total_with_speech = sum(
            1 for config_results in self.results.values()
            for r in config_results.values()
            if r.has_speech
        )
        total_without = sum(
            1 for config_results in self.results.values()
            for r in config_results.values()
            if not r.has_speech
        )
        
        if total_with_speech + total_without > 0:
            sizes = [total_with_speech, total_without]
            labels = [f'With Speech\n({total_with_speech})', f'Without Speech\n({total_without})']
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0)
            
            ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.set_title('Overall Speech Detection Rate', fontweight='bold')
        
        # Plot 3: Configuration performance radar-like comparison
        ax3 = axes[1, 0]
        config_names = list(TEST_CONFIGURATIONS.keys())
        metrics = {
            'Detection Rate': [],
            'Avg Probability': [],
            'Consistency': [],
        }
        
        for config_name in config_names:
            if config_name in self.results:
                results = list(self.results[config_name].values())
                metrics['Detection Rate'].append(
                    np.mean([r.has_speech for r in results]) * 100
                )
                metrics['Avg Probability'].append(
                    np.mean([r.total_speech_probability for r in results]) * 100
                )
                # Consistency: lower std = more consistent
                probs = [r.total_speech_probability for r in results]
                consistency = 100 - (np.std(probs) * 100) if probs else 0
                metrics['Consistency'].append(max(0, consistency))
            else:
                for metric in metrics.values():
                    metric.append(0)
        
        x = np.arange(len(config_names))
        width = 0.25
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            bars = ax3.bar(x + i * width, values, width, label=metric_name, alpha=0.8)
        
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Score (%)')
        ax3.set_title('Configuration Performance Comparison', fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([c.replace('_', ' ').title() for c in config_names], 
                           rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Processing time comparison
        ax4 = axes[1, 1]
        config_times = {}
        for config_name, config_results in self.results.items():
            if config_results:
                avg_time = np.mean([r.processing_time for r in config_results.values()])
                config_times[config_name] = avg_time
        
        if config_times:
            names = list(config_times.keys())
            times = list(config_times.values())
            
            bars = ax4.barh(range(len(names)), times, 
                           color=plt.cm.viridis(np.linspace(0.2, 0.9, len(names))))
            ax4.set_yticks(range(len(names)))
            ax4.set_yticklabels([n.replace('_', ' ').title() for n in names])
            ax4.set_xlabel('Average Processing Time (seconds)')
            ax4.set_title('Processing Time by Configuration', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add time labels
            for bar, time_val in zip(bars, times):
                ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{time_val:.2f}s', va='center')
        
        plt.tight_layout()
        viz_path = output_dir / "overall_analysis.png"
        fig.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        console.print(f"[green]✓ Saved overall visualizations: {viz_path}[/green]")
    
    def generate_file_tree(self) -> None:
        """Generate a tree view of all generated files."""
        tree = Tree(f"[bold cyan]📁 {self.output_base.name}[/bold cyan]")
        
        for config_name in sorted(TEST_CONFIGURATIONS.keys()):
            config_dir = self.output_base / config_name
            if config_dir.exists():
                config_node = tree.add(f"[yellow]⚙ {config_name}[/yellow]")
                
                # Add file results
                for wav_dir in sorted(config_dir.iterdir()):
                    if wav_dir.is_dir():
                        file_node = config_node.add(f"[cyan]🎵 {wav_dir.name}[/cyan]")
                        for file in sorted(wav_dir.glob("*")):
                            icon = "📊" if file.suffix == ".png" else "📄"
                            file_node.add(f"[dim]{icon} {file.name}[/dim]")
                
                # Add summary files
                for summary_file in sorted(config_dir.glob("*_summary.json")):
                    config_node.add(f"[green]📋 {summary_file.name}[/green]")
        
        # Add analysis directories
        for analysis_dir in ["comparative_analysis", "summary", "overall_visualizations"]:
            dir_path = self.output_base / analysis_dir
            if dir_path.exists():
                analysis_node = tree.add(f"[magenta]📈 {analysis_dir.replace('_', ' ').title()}[/magenta]")
                for file in sorted(dir_path.iterdir()):
                    icon = "📊" if file.suffix == ".png" else "📄"
                    analysis_node.add(f"[dim]{icon} {file.name}[/dim]")
        
        console.print(tree)


def main():
    """Main demo execution."""
    console.print(Panel.fit(
        "[bold yellow]🎙️ SpeechChecker Demo Suite[/bold yellow]\n"
        "[dim]Testing Zipformer audio tagging with multiple configurations[/dim]",
        border_style="yellow"
    ))
    
    try:
        # Initialize demo runner
        demo = DemoRunner()
        
        # Display test files
        test_files = demo.get_test_files()
        
        if not test_files:
            console.print("[red]No test WAV files found![/red]")
            console.print(
                "Please download the Zipformer model from:\n"
                "https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models"
            )
            return
        
        # Show what we're about to do
        console.print("\n[bold]Test Files:[/bold]")
        for i, wav_file in enumerate(test_files, 1):
            console.print(f"  {i}. {wav_file.name}")
        
        console.print(f"\n[bold]Configurations to Test:[/bold]")
        for config_name, config in TEST_CONFIGURATIONS.items():
            speech_types = [SPEECH_INDICES[i] for i in config["speech_indices"]]
            console.print(f"  • [cyan]{config_name}[/cyan]: {config['description']}")
            console.print(f"    Threshold: {config['threshold']}, Types: {', '.join(speech_types[:3])}{'...' if len(speech_types) > 3 else ''}")
        
        # Confirm before running
        console.print("\n[yellow]Starting demo processing...[/yellow]\n")
        
        # Run all configurations
        demo.run_all_configurations(test_files)
        
        # Generate reports
        demo.generate_reports()
        
        # Show file tree
        console.print("\n[bold cyan]📁 Generated File Structure:[/bold cyan]")
        demo.generate_file_tree()
        
        # Final summary
        elapsed = (datetime.now() - demo.start_time).total_seconds()
        console.print(Panel.fit(
            f"[bold green]✅ Demo Complete![/bold green]\n"
            f"[dim]Total time: {elapsed:.1f}s | "
            f"Files processed: {len(test_files)} | "
            f"Configurations: {len(TEST_CONFIGURATIONS)} | "
            f"Total runs: {len(test_files) * len(TEST_CONFIGURATIONS)}[/dim]\n"
            f"[dim]All results saved to: {DEMO_OUTPUT_BASE}[/dim]",
            border_style="green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        log.exception("Demo failed")
        raise


if __name__ == "__main__":
    main()
