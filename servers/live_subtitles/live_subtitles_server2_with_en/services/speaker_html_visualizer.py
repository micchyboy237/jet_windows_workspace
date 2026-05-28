"""HTML visualization module for speaker plots."""

import base64
import io
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

# Import your visualizer
try:
    from services.speaker_visualizer import SpeakerVisualizer
except ImportError:
    from speaker_visualizer import SpeakerVisualizer


class SpeakerHTMLVisualizer:
    """Generate HTML reports with embedded speaker visualizations."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.visualizer = SpeakerVisualizer(save_dir=None)  # Don't auto-save
        self.figures: Dict[str, str] = {}  # Store base64 encoded figures
    
    def _figure_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _create_figure(self, fig: Optional[plt.Figure]) -> Optional[str]:
        """Convert figure to base64 if it exists."""
        if fig is not None:
            return self._figure_to_base64(fig)
        return None
    
    def generate_all_plots(self, labeler) -> Dict[str, str]:
        """Generate all plots and return as base64 encoded strings."""
        self.figures = {}
        
        # Generate each plot
        self.figures['centroids_pca'] = self._create_figure(
            self.visualizer.plot_centroids_2d(labeler, method='pca')
        )
        
        if len(labeler.known_speakers) >= 3:
            self.figures['centroids_tsne'] = self._create_figure(
                self.visualizer.plot_centroids_2d(labeler, method='tsne')
            )
        else:
            self.figures['centroids_tsne'] = None
        
        self.figures['heatmap'] = self._create_figure(
            self.visualizer.plot_similarity_heatmap(labeler)
        )
        
        self.figures['timeline'] = self._create_figure(
            self.visualizer.plot_timeline(labeler)
        )
        
        self.figures['dashboard'] = self._create_figure(
            self.visualizer.plot_dashboard(labeler)
        )
        
        return self.figures
    
    def get_dashboard_html(self, labeler) -> str:
        """Generate complete HTML dashboard page."""
        self.generate_all_plots(labeler)
        
        # Get speaker data
        speakers_info = labeler.get_all_speakers_info()
        health_status = labeler.get_health_status()
        similarity_matrix = labeler.get_speaker_similarity_matrix()
        
        html_template = Template("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Speaker Diarization Dashboard</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                }
                
                .header {
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    margin-bottom: 20px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }
                
                .header h1 {
                    color: #333;
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }
                
                .header .subtitle {
                    color: #666;
                    font-size: 1.1em;
                }
                
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                
                .stat-card {
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease;
                }
                
                .stat-card:hover {
                    transform: translateY(-5px);
                }
                
                .stat-card .stat-value {
                    font-size: 3em;
                    font-weight: bold;
                    color: #667eea;
                    margin: 10px 0;
                }
                
                .stat-card .stat-label {
                    color: #666;
                    font-size: 1.1em;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                
                .stat-card .stat-detail {
                    color: #999;
                    font-size: 0.9em;
                    margin-top: 5px;
                }
                
                .alert-list {
                    list-style: none;
                    margin-top: 10px;
                }
                
                .alert-list li {
                    padding: 8px 12px;
                    margin: 5px 0;
                    border-radius: 8px;
                    font-size: 0.9em;
                }
                
                .alert-list li.healthy {
                    background: #d4edda;
                    color: #155724;
                }
                
                .alert-list li.warning {
                    background: #fff3cd;
                    color: #856404;
                }
                
                .plot-section {
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    margin-bottom: 20px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }
                
                .plot-section h2 {
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 1.8em;
                }
                
                .plot-section img {
                    width: 100%;
                    height: auto;
                    border-radius: 10px;
                }
                
                .plot-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 20px;
                }
                
                .table-container {
                    overflow-x: auto;
                    margin-top: 20px;
                }
                
                table {
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    border-radius: 10px;
                    overflow: hidden;
                }
                
                thead {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                
                th {
                    padding: 15px;
                    text-align: left;
                    font-weight: 600;
                    text-transform: uppercase;
                    font-size: 0.9em;
                    letter-spacing: 1px;
                }
                
                td {
                    padding: 12px 15px;
                    border-bottom: 1px solid #f0f0f0;
                }
                
                tr:hover {
                    background: #f8f9fa;
                }
                
                .quality-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                
                .quality-excellent {
                    background: #28a745;
                    box-shadow: 0 0 10px rgba(40, 167, 69, 0.5);
                }
                
                .quality-good {
                    background: #ffc107;
                    box-shadow: 0 0 10px rgba(255, 193, 7, 0.5);
                }
                
                .quality-poor {
                    background: #dc3545;
                    box-shadow: 0 0 10px rgba(220, 53, 69, 0.5);
                }
                
                .controls {
                    display: flex;
                    gap: 15px;
                    margin-top: 20px;
                    flex-wrap: wrap;
                }
                
                .btn {
                    padding: 12px 24px;
                    border: none;
                    border-radius: 8px;
                    font-size: 1em;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                
                .btn-primary {
                    background: #667eea;
                    color: white;
                }
                
                .btn-primary:hover {
                    background: #5a67d8;
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                }
                
                .btn-danger {
                    background: #dc3545;
                    color: white;
                }
                
                .btn-danger:hover {
                    background: #c82333;
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(220, 53, 69, 0.4);
                }
                
                .btn-warning {
                    background: #ffc107;
                    color: #333;
                }
                
                .btn-warning:hover {
                    background: #e0a800;
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(255, 193, 7, 0.4);
                }
                
                .similarity-cell {
                    font-weight: 600;
                    text-align: center;
                }
                
                .similarity-high {
                    background: #d4edda;
                    color: #155724;
                }
                
                .similarity-medium {
                    background: #fff3cd;
                    color: #856404;
                }
                
                .similarity-low {
                    background: #f8d7da;
                    color: #721c24;
                }
                
                @media (max-width: 768px) {
                    .plot-grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .stats-grid {
                        grid-template-columns: 1fr;
                    }
                    
                    .header h1 {
                        font-size: 1.8em;
                    }
                }
                
                .loading {
                    display: none;
                    text-align: center;
                    padding: 40px;
                }
                
                .loading.active {
                    display: block;
                }
                
                .spinner {
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #667eea;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Header -->
                <div class="header">
                    <h1>🎙️ Speaker Diarization Dashboard</h1>
                    <p class="subtitle">Real-time speaker analysis and visualization</p>
                    
                    <div class="controls">
                        <button class="btn btn-primary" onclick="refreshDashboard()">
                            🔄 Refresh
                        </button>
                        <button class="btn btn-warning" onclick="consolidateSpeakers()">
                            🔗 Consolidate Speakers
                        </button>
                        <button class="btn btn-danger" onclick="resetSpeakers()">
                            🗑️ Reset All
                        </button>
                    </div>
                </div>
                
                <!-- Statistics Cards -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Speakers</div>
                        <div class="stat-value">{{ health_status.total_speakers }}</div>
                        <div class="stat-detail">Active speakers tracked</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Mature Speakers</div>
                        <div class="stat-value">{{ health_status.mature_speakers }}</div>
                        <div class="stat-detail">Reliable centroids</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Young Speakers</div>
                        <div class="stat-value">{{ health_status.young_speakers }}</div>
                        <div class="stat-detail">Still learning</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Health Status</div>
                        <div class="stat-value">
                            {% if health_status.alerts[0].startswith('✅') %}
                                ✅
                            {% else %}
                                ⚠️
                            {% endif %}
                        </div>
                        <div class="stat-detail">
                            <ul class="alert-list">
                                {% for alert in health_status.alerts %}
                                    <li class="{{ 'healthy' if '✅' in alert else 'warning' }}">
                                        {{ alert }}
                                    </li>
                                {% endfor %}
                            </ul>
                        </div>
                    </div>
                </div>
                
                <!-- Speaker Details Table -->
                <div class="plot-section">
                    <h2>📊 Speaker Details</h2>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Speaker</th>
                                    <th>Segments</th>
                                    <th>First Seen</th>
                                    <th>Last Seen</th>
                                    <th>Duration</th>
                                    <th>Quality</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for label, info in speakers_info.items() %}
                                <tr>
                                    <td><strong>{{ label }}</strong></td>
                                    <td>{{ info.segment_count }}</td>
                                    <td>{{ "%.1f"|format(info.first_seen) }}s</td>
                                    <td>{{ "%.1f"|format(info.last_seen) }}s</td>
                                    <td>{{ "%.1f"|format(info.active_duration) }}s</td>
                                    <td>
                                        {% if info.centroid_quality >= 0.8 %}
                                            <span class="quality-indicator quality-excellent"></span>
                                            Excellent ({{ "%.2f"|format(info.centroid_quality) }})
                                        {% elif info.centroid_quality >= 0.6 %}
                                            <span class="quality-indicator quality-good"></span>
                                            Good ({{ "%.2f"|format(info.centroid_quality) }})
                                        {% else %}
                                            <span class="quality-indicator quality-poor"></span>
                                            Poor ({{ "%.2f"|format(info.centroid_quality) }})
                                        {% endif %}
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Similarity Matrix -->
                {% if similarity_matrix.labels|length > 1 %}
                <div class="plot-section">
                    <h2>🔗 Speaker Similarity Matrix</h2>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Speaker</th>
                                    {% for label in similarity_matrix.labels %}
                                    <th>{{ label }}</th>
                                    {% endfor %}
                                </tr>
                            </thead>
                            <tbody>
                                {% for i in range(similarity_matrix.labels|length) %}
                                <tr>
                                    <td><strong>{{ similarity_matrix.labels[i] }}</strong></td>
                                    {% for j in range(similarity_matrix.labels|length) %}
                                        {% set sim = similarity_matrix.similarities[i][j] %}
                                        <td class="similarity-cell 
                                            {% if sim >= 0.8 %}similarity-high
                                            {% elif sim >= 0.5 %}similarity-medium
                                            {% else %}similarity-low{% endif %}">
                                            {{ "%.3f"|format(sim) }}
                                        </td>
                                    {% endfor %}
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                {% endif %}
                
                <!-- Plots -->
                {% if figures.centroids_pca %}
                <div class="plot-section">
                    <h2>🎯 Speaker Embeddings (PCA)</h2>
                    <img src="data:image/png;base64,{{ figures.centroids_pca }}" alt="PCA Plot">
                </div>
                {% endif %}
                
                {% if figures.centroids_tsne %}
                <div class="plot-section">
                    <h2>🎯 Speaker Embeddings (t-SNE)</h2>
                    <img src="data:image/png;base64,{{ figures.centroids_tsne }}" alt="t-SNE Plot">
                </div>
                {% endif %}
                
                <div class="plot-grid">
                    {% if figures.heatmap %}
                    <div class="plot-section">
                        <h2>🔥 Similarity Heatmap</h2>
                        <img src="data:image/png;base64,{{ figures.heatmap }}" alt="Heatmap">
                    </div>
                    {% endif %}
                    
                    {% if figures.timeline %}
                    <div class="plot-section">
                        <h2>⏱️ Activity Timeline</h2>
                        <img src="data:image/png;base64,{{ figures.timeline }}" alt="Timeline">
                    </div>
                    {% endif %}
                </div>
                
                {% if figures.dashboard %}
                <div class="plot-section">
                    <h2>📈 Dashboard Overview</h2>
                    <img src="data:image/png;base64,{{ figures.dashboard }}" alt="Dashboard">
                </div>
                {% endif %}
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing...</p>
            </div>
            
            <script>
                function refreshDashboard() {
                    document.getElementById('loading').classList.add('active');
                    location.reload();
                }
                
                async function consolidateSpeakers() {
                    if (!confirm('Consolidate similar speakers? This will merge speakers with high similarity.')) {
                        return;
                    }
                    
                    const threshold = prompt('Enter consolidation threshold (0.0-1.0):', '0.85');
                    if (!threshold) return;
                    
                    document.getElementById('loading').classList.add('active');
                    
                    try {
                        const formData = new FormData();
                        formData.append('threshold', threshold);
                        formData.append('dry_run', 'false');
                        
                        const response = await fetch('/speakers/consolidate', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            alert(`Consolidation complete! Merged ${result.merges_performed.length} speaker pairs.`);
                            location.reload();
                        } else {
                            alert('Consolidation failed: ' + JSON.stringify(result));
                        }
                    } catch (error) {
                        alert('Error: ' + error.message);
                    } finally {
                        document.getElementById('loading').classList.remove('active');
                    }
                }
                
                async function resetSpeakers() {
                    if (!confirm('Are you sure you want to reset ALL speaker data? This cannot be undone.')) {
                        return;
                    }
                    
                    document.getElementById('loading').classList.add('active');
                    
                    try {
                        const response = await fetch('/speakers/reset', {
                            method: 'POST'
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            alert('All speaker data has been reset.');
                            location.reload();
                        } else {
                            alert('Reset failed: ' + JSON.stringify(result));
                        }
                    } catch (error) {
                        alert('Error: ' + error.message);
                    } finally {
                        document.getElementById('loading').classList.remove('active');
                    }
                }
            </script>
        </body>
        </html>
        """)
        
        return html_template.render(
            health_status=health_status,
            speakers_info=speakers_info,
            similarity_matrix=similarity_matrix,
            figures=self.figures,
        )
    
    def get_plots_only_html(self, labeler, plots: Optional[list] = None) -> str:
        """Generate HTML with just selected plots (no dashboard).
        
        Parameters
        ----------
        labeler : SegmentSpeakerLabeler
            The speaker labeler instance.
        plots : list, optional
            List of plot names to include. Options: 'pca', 'tsne', 'heatmap', 
            'timeline', 'dashboard'. If None, includes all.
        """
        if plots is None:
            plots = ['pca', 'tsne', 'heatmap', 'timeline', 'dashboard']
        
        plot_methods = {
            'pca': ('centroids_pca', lambda: self.visualizer.plot_centroids_2d(labeler, method='pca')),
            'tsne': ('centroids_tsne', lambda: self.visualizer.plot_centroids_2d(labeler, method='tsne')),
            'heatmap': ('heatmap', lambda: self.visualizer.plot_similarity_heatmap(labeler)),
            'timeline': ('timeline', lambda: self.visualizer.plot_timeline(labeler)),
            'dashboard': ('dashboard', lambda: self.visualizer.plot_dashboard(labeler)),
        }
        
        self.figures = {}
        for plot_name in plots:
            if plot_name in plot_methods:
                key, func = plot_methods[plot_name]
                self.figures[key] = self._create_figure(func())
        
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Speaker Analysis Plots</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 20px;
                    background: #f5f5f5;
                }
                .plot-container {
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .plot-container h2 {
                    color: #333;
                    margin-bottom: 15px;
                }
                .plot-container img {
                    width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            {% for plot_key, plot_data in figures.items() %}
                {% if plot_data %}
                <div class="plot-container">
                    <h2>{{ plot_key|replace('_', ' ')|title }}</h2>
                    <img src="data:image/png;base64,{{ plot_data }}" alt="{{ plot_key }}">
                </div>
                {% endif %}
            {% endfor %}
        </body>
        </html>
        """)
        
        return html_template.render(figures=self.figures)