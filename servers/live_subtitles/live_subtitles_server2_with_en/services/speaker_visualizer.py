"""Visualization class for SegmentSpeakerLabeler centroids and statistics."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

console = Console()


class SpeakerVisualizer:
    """Visualize speaker centroids, similarities, timelines, and statistics.
    
    Parameters
    ----------
    save_dir : str, optional
        Directory to save plots. If provided, plots will be automatically saved
        with timestamps. If None, plots are only displayed.
    dpi : int
        DPI for saved figures (default: 150).
    style : str
        Matplotlib style to use (default: 'seaborn-v0_8-darkgrid').
    figsize_scale : float
        Scale factor for figure sizes (default: 1.0).
    """
    
    def __init__(
        self,
        save_dir: Optional[str] = None,
        dpi: int = 150,
        style: str = 'seaborn-v0_8-darkgrid',
        figsize_scale: float = 1.0,
    ):
        self.save_dir = save_dir
        self.dpi = dpi
        self.figsize_scale = figsize_scale
        
        # Set up style
        try:
            plt.style.use(style)
        except Exception:
            console.print(f"[yellow]Style '{style}' not found, using default[/]")
            plt.style.use('default')
        
        # Create save directory if specified
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def _get_save_path(self, name: str, extension: str = "png") -> Optional[str]:
        """Generate a save path with timestamp."""
        if not self.save_dir:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{name}.{extension}"
        return os.path.join(self.save_dir, filename)
    
    def _save_or_show(self, fig: plt.Figure, name: str) -> None:
        """Save figure if save_dir is set, otherwise display it."""
        if self.save_dir:
            path = self._get_save_path(name)
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            console.print(f"[green]✓ Saved: {path}[/]")
            plt.close(fig)
        else:
            plt.show()
    
    def _collect_centroids(
        self, labeler
    ) -> Tuple[np.ndarray, List[str], List[int], List[float]]:
        """Collect valid centroids and metadata from labeler."""
        centroids = []
        labels = []
        segment_counts = []
        qualities = []
        
        for label in labeler.known_speakers:
            info = labeler.get_speaker_info(label)
            if info.get('centroid_coordinates') is not None:
                centroids.append(info['centroid_coordinates'])
                labels.append(label)
                segment_counts.append(info['segment_count'])
                qualities.append(info['centroid_quality'])
        
        if centroids:
            centroids_array = np.array(centroids).squeeze()
            # Remove extra dimension if present (1, N, D) -> (N, D)
            if centroids_array.ndim == 3:
                centroids_array = centroids_array[:, 0, :]
        else:
            centroids_array = np.array([])
        
        return centroids_array, labels, segment_counts, qualities
    
    def plot_centroids_2d(
        self,
        labeler,
        method: str = 'pca',
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
        show_labels: bool = True,
        colormap: str = 'viridis',
        random_state: int = 42,
    ) -> plt.Figure:
        """Plot all speaker centroids in 2D space using dimensionality reduction.
        
        Parameters
        ----------
        labeler : SegmentSpeakerLabeler
            The speaker labeler instance to visualize.
        method : str
            Dimensionality reduction method: 'pca' or 'tsne'.
        figsize : tuple
            Figure size (width, height).
        title : str, optional
            Custom plot title.
        show_labels : bool
            Whether to show speaker labels on points.
        colormap : str
            Matplotlib colormap for centroid quality.
        random_state : int
            Random seed for reproducibility.
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        centroids_array, labels, segment_counts, qualities = self._collect_centroids(labeler)
        
        if len(centroids_array) < 2:
            console.print("[yellow]Need at least 2 speakers with valid centroids[/]")
            return None
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=random_state)
        elif method == 'tsne':
            perplexity = min(30, len(centroids_array) - 1)
            reducer = TSNE(
                n_components=2, 
                random_state=random_state, 
                perplexity=perplexity
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
        
        centroids_2d = reducer.fit_transform(centroids_array)
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(figsize[0] * self.figsize_scale, 
                    figsize[1] * self.figsize_scale)
        )
        
        # Plot points
        sizes = np.array(segment_counts) * 50
        scatter = ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            s=sizes,
            c=qualities,
            cmap=colormap,
            alpha=0.7,
            edgecolors='black',
            linewidth=1,
            zorder=5,
        )
        
        # Add labels
        if show_labels:
            for i, label in enumerate(labels):
                ax.annotate(
                    f"{label}\n({segment_counts[i]} segs)",
                    (centroids_2d[i, 0], centroids_2d[i, 1]),
                    xytext=(7, 7),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.9,
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='gray'
                    )
                )
        
        # Colorbar and labels
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Centroid Quality', fontsize=11)
        
        if title is None:
            title = f'Speaker Centroids Visualization ({method.upper()})'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=11)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_or_show(fig, f"centroids_2d_{method}")
        return fig
    
    def plot_similarity_heatmap(
        self,
        labeler,
        figsize: Tuple[int, int] = (10, 8),
        annotate: bool = True,
        cmap: str = 'coolwarm',
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a heatmap of pairwise speaker similarities.
        
        Parameters
        ----------
        labeler : SegmentSpeakerLabeler
            The speaker labeler instance.
        figsize : tuple
            Figure size (width, height).
        annotate : bool
            Whether to show similarity values in cells.
        cmap : str
            Colormap for the heatmap.
        title : str, optional
            Custom plot title.
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        matrix_data = labeler.get_speaker_similarity_matrix()
        
        if len(matrix_data['labels']) < 2:
            console.print("[yellow]Need at least 2 speakers for similarity matrix[/]")
            return None
        
        fig, ax = plt.subplots(
            figsize=(figsize[0] * self.figsize_scale, 
                    figsize[1] * self.figsize_scale)
        )
        
        sns.heatmap(
            matrix_data['similarities'],
            xticklabels=matrix_data['labels'],
            yticklabels=matrix_data['labels'],
            annot=annotate,
            fmt='.3f',
            cmap=cmap,
            center=0.5,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Cosine Similarity'},
            ax=ax,
        )
        
        if title is None:
            title = 'Speaker Similarity Matrix'
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        self._save_or_show(fig, "similarity_heatmap")
        return fig
    
    def plot_timeline(
        self,
        labeler,
        figsize: Tuple[int, int] = (14, 7),
        title: Optional[str] = None,
        color_palette: str = 'tab20',
    ) -> plt.Figure:
        """Plot speaker activity timeline.
        
        Parameters
        ----------
        labeler : SegmentSpeakerLabeler
            The speaker labeler instance.
        figsize : tuple
            Figure size (width, height).
        title : str, optional
            Custom plot title.
        color_palette : str
            Matplotlib colormap name for speaker colors.
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        speakers_info = labeler.get_all_speakers_info()
        
        if not speakers_info:
            console.print("[yellow]No speakers to plot timeline[/]")
            return None
        
        fig, ax = plt.subplots(
            figsize=(figsize[0] * self.figsize_scale, 
                    figsize[1] * self.figsize_scale)
        )
        
        colors = plt.cm.get_cmap(color_palette)(
            np.linspace(0, 1, max(len(speakers_info), 1))
        )
        
        # Sort speakers by first appearance
        sorted_speakers = sorted(
            speakers_info.items(),
            key=lambda x: x[1]['first_seen'] if x[1]['first_seen'] is not None else 0
        )
        
        for i, (label, info) in enumerate(sorted_speakers):
            first_seen = info['first_seen'] if info['first_seen'] is not None else 0.0
            last_seen = info['last_seen']
            duration = last_seen - first_seen
            
            if duration > 0:
                ax.barh(
                    i,
                    duration,
                    left=first_seen,
                    height=0.7,
                    color=colors[i % len(colors)],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5,
                )
            
            # Add segment count annotation
            ax.text(
                last_seen + max(duration * 0.02, 0.1),
                i,
                f"{label} ({info['segment_count']} segs, "
                f"Q: {info['centroid_quality']:.2f})",
                va='center',
                fontsize=9,
            )
        
        ax.set_yticks(range(len(sorted_speakers)))
        ax.set_yticklabels([label for label, _ in sorted_speakers])
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        if title is None:
            title = 'Speaker Activity Timeline'
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Invert y-axis to show first speaker at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        self._save_or_show(fig, "timeline")
        return fig
    
    def plot_dashboard(
        self,
        labeler,
        figsize: Tuple[int, int] = (16, 10),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Create a comprehensive dashboard of speaker statistics.
        
        Parameters
        ----------
        labeler : SegmentSpeakerLabeler
            The speaker labeler instance.
        figsize : tuple
            Figure size (width, height).
        title : str, optional
            Custom overall title.
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        speakers_info = labeler.get_all_speakers_info()
        
        if not speakers_info:
            console.print("[yellow]No speakers to create dashboard[/]")
            return None
        
        fig, axes = plt.subplots(
            2, 2,
            figsize=(figsize[0] * self.figsize_scale, 
                    figsize[1] * self.figsize_scale)
        )
        
        labels = list(speakers_info.keys())
        counts = [info['segment_count'] for info in speakers_info.values()]
        durations = [info['active_duration'] for info in speakers_info.values()]
        qualities = [info['centroid_quality'] for info in speakers_info.values()]
        
        # Color mapping
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(labels)))
        
        # 1. Segment count bar chart
        ax1 = axes[0, 0]
        bars1 = ax1.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_title('Segments per Speaker', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Segment Count')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars1, counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha='center',
                va='bottom',
                fontsize=9,
            )
        
        # 2. Active duration
        ax2 = axes[0, 1]
        bars2 = ax2.bar(labels, durations, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_title('Active Duration per Speaker', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Duration (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, dur in zip(bars2, durations):
            if dur > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(durations) * 0.01,
                    f'{dur:.1f}s',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )
        
        # 3. Centroid quality
        ax3 = axes[1, 0]
        bars3 = ax3.bar(labels, qualities, color=colors, edgecolor='black', linewidth=0.5)
        ax3.set_title('Centroid Quality', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Quality Score')
        ax3.set_ylim(0, 1.1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add threshold lines
        ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Excellent')
        ax3.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Good')
        ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Poor')
        ax3.legend(loc='upper right', fontsize=8)
        
        for bar, qual in zip(bars3, qualities):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{qual:.2f}',
                ha='center',
                va='bottom',
                fontsize=9,
            )
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        mature_threshold = 5
        young_threshold = 2
        
        mature_count = sum(1 for c in counts if c >= mature_threshold)
        young_count = sum(1 for c in counts if c <= young_threshold)
        middle_count = len(counts) - mature_count - young_count
        
        summary_text = f"""
        ╔══════════════════════════╗
        ║  SPEAKER STATISTICS      ║
        ╚══════════════════════════╝
        
        📊 Overview
        ─────────────────────────
        Total Speakers:      {len(labels):>4d}
        Total Segments:      {sum(counts):>4d}
        Avg Seg/Speaker:     {np.mean(counts):>6.1f}
        Avg Duration:        {np.mean(durations):>6.1f}s
        Avg Quality:         {np.mean(qualities):>6.2f}
        
        📈 Categories
        ─────────────────────────
        Mature (≥{mature_threshold} segs):    {mature_count:>4d}
        Growing ({young_threshold+1}-{mature_threshold-1}):    {middle_count:>4d}
        Young (≤{young_threshold} segs):     {young_count:>4d}
        
        ⏱️ Temporal
        ─────────────────────────
        Total Span:          {max(durations):>6.1f}s
        Max Duration:        {max(durations):>6.1f}s
        Min Duration:        {min(durations):>6.1f}s
        
        🔍 Quality
        ─────────────────────────
        Excellent (≥0.8):    {sum(1 for q in qualities if q >= 0.8):>4d}
        Good (≥0.6):         {sum(1 for q in qualities if q >= 0.6):>4d}
        Poor (<0.3):         {sum(1 for q in qualities if q < 0.3):>4d}
        """
        
        ax4.text(
            0.05, 0.95, summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            family='monospace',
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='lightgray',
                alpha=0.2
            )
        )
        
        if title:
            fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        else:
            fig.suptitle('Speaker Labeling Dashboard', fontsize=15, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        self._save_or_show(fig, "dashboard")
        return fig
    
    def plot_3d_interactive(
        self,
        labeler,
        method: str = 'pca',
        title: Optional[str] = None,
        random_state: int = 42,
    ):
        """Create an interactive 3D plot using plotly (requires plotly).
        
        Parameters
        ----------
        labeler : SegmentSpeakerLabeler
            The speaker labeler instance.
        method : str
            Dimensionality reduction: 'pca' or 'tsne'.
        title : str, optional
            Custom plot title.
        random_state : int
            Random seed.
            
        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        try:
            import plotly.express as px
            import pandas as pd
        except ImportError:
            console.print("[red]Plotly is required for interactive 3D plots. "
                         "Install with: pip install plotly[/]")
            return None
        
        centroids_array, labels, segment_counts, qualities = self._collect_centroids(labeler)
        
        if len(centroids_array) < 3:
            console.print("[yellow]Need at least 3 speakers for 3D plot[/]")
            return None
        
        # Reduce to 3D
        if method == 'pca':
            reducer = PCA(n_components=3, random_state=random_state)
        elif method == 'tsne':
            perplexity = min(30, len(centroids_array) - 1)
            reducer = TSNE(
                n_components=3,
                random_state=random_state,
                perplexity=perplexity
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        centroids_3d = reducer.fit_transform(centroids_array)
        
        # Create DataFrame
        df = pd.DataFrame({
            'PC1': centroids_3d[:, 0],
            'PC2': centroids_3d[:, 1],
            'PC3': centroids_3d[:, 2],
            'Speaker': labels,
            'Segments': segment_counts,
            'Quality': qualities,
            'Size': [max(c * 8, 5) for c in segment_counts],
        })
        
        fig = px.scatter_3d(
            df,
            x='PC1', y='PC2', z='PC3',
            text='Speaker',
            size='Size',
            color='Quality',
            hover_data=['Speaker', 'Segments', 'Quality'],
            color_continuous_scale='viridis',
            title=title or f'Speaker Embeddings in 3D ({method.upper()})',
        )
        
        fig.update_traces(
            marker=dict(
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            textposition='top center',
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f'{method.upper()} 1',
                yaxis_title=f'{method.upper()} 2',
                zaxis_title=f'{method.upper()} 3',
            )
        )
        
        if self.save_dir:
            path = self._get_save_path(f"centroids_3d_{method}", "html")
            fig.write_html(path)
            console.print(f"[green]✓ Saved interactive plot: {path}[/]")
        else:
            fig.show()
        
        return fig
    
    def plot_all(
        self,
        labeler,
        include_3d: bool = False,
    ) -> Dict[str, plt.Figure]:
        """Generate all standard plots at once.
        
        Parameters
        ----------
        labeler : SegmentSpeakerLabeler
            The speaker labeler instance.
        include_3d : bool
            Whether to include interactive 3D plot (requires plotly).
            
        Returns
        -------
        dict
            Dictionary of plot names to figures.
        """
        figures = {}
        
        console.print("[bold]Generating speaker visualizations...[/]")
        
        # 2D centroids (PCA)
        console.print("  → PCA centroids plot...")
        figures['centroids_pca'] = self.plot_centroids_2d(labeler, method='pca')
        
        # 2D centroids (t-SNE)
        if len(labeler.known_speakers) >= 3:
            console.print("  → t-SNE centroids plot...")
            figures['centroids_tsne'] = self.plot_centroids_2d(labeler, method='tsne')
        
        # Similarity heatmap
        console.print("  → Similarity heatmap...")
        figures['heatmap'] = self.plot_similarity_heatmap(labeler)
        
        # Timeline
        console.print("  → Activity timeline...")
        figures['timeline'] = self.plot_timeline(labeler)
        
        # Dashboard
        console.print("  → Dashboard...")
        figures['dashboard'] = self.plot_dashboard(labeler)
        
        # 3D interactive (optional)
        if include_3d:
            console.print("  → Interactive 3D plot...")
            figures['3d_interactive'] = self.plot_3d_interactive(labeler)
        
        console.print(f"[bold green]✓ Generated {len(figures)} plots[/]")
        return figures


# Example usage and integration with SegmentSpeakerLabeler
if __name__ == "__main__":
    # Example with mock data
    from unittest.mock import MagicMock
    
    # Create a mock labeler for demonstration
    # In real usage, you'd pass your actual SegmentSpeakerLabeler instance
    labeler = MagicMock()
    
    # Setup mock data
    labeler.known_speakers = ['SPEAKER_01', 'SPEAKER_02', 'SPEAKER_03']
    
    def mock_get_speaker_info(label):
        info_map = {
            'SPEAKER_01': {
                'centroid_coordinates': [[0.1, 0.2, 0.3, 0.4]],
                'segment_count': 15,
                'centroid_quality': 1.0,
                'first_seen': 0.5,
                'last_seen': 25.0,
                'active_duration': 24.5,
            },
            'SPEAKER_02': {
                'centroid_coordinates': [[0.8, 0.7, 0.6, 0.5]],
                'segment_count': 8,
                'centroid_quality': 0.8,
                'first_seen': 2.0,
                'last_seen': 20.0,
                'active_duration': 18.0,
            },
            'SPEAKER_03': {
                'centroid_coordinates': [[0.5, 0.5, 0.5, 0.5]],
                'segment_count': 2,
                'centroid_quality': 0.3,
                'first_seen': 10.0,
                'last_seen': 15.0,
                'active_duration': 5.0,
            },
        }
        return info_map.get(label)
    
    labeler.get_speaker_info = mock_get_speaker_info
    
    def mock_get_all_speakers_info():
        return {label: mock_get_speaker_info(label) for label in labeler.known_speakers}
    
    labeler.get_all_speakers_info = mock_get_all_speakers_info
    
    def mock_similarity_matrix():
        return {
            'labels': ['SPEAKER_01', 'SPEAKER_02', 'SPEAKER_03'],
            'similarities': [
                [1.0, 0.15, 0.45],
                [0.15, 1.0, 0.62],
                [0.45, 0.62, 1.0],
            ],
            'segment_counts': [15, 8, 2],
        }
    
    labeler.get_speaker_similarity_matrix = mock_similarity_matrix
    
    # Create visualizer with save directory
    visualizer = SpeakerVisualizer(
        save_dir="./speaker_plots",
        dpi=150,
    )
    
    # Generate all plots
    figures = visualizer.plot_all(labeler, include_3d=False)
    
    # Or generate individual plots
    # visualizer.plot_centroids_2d(labeler, method='pca')
    # visualizer.plot_similarity_heatmap(labeler)
    # visualizer.plot_timeline(labeler)
    # visualizer.plot_dashboard(labeler)