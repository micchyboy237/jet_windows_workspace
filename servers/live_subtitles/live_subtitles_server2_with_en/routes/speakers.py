"""
Speaker management routes.
"""
from datetime import datetime
from typing import Dict, Optional
from fastapi import APIRouter, Form, HTTPException
from core.state import (
    get_speaker_labeler,
    set_current_speaker,
    set_last_speaker_change_time,
    get_context_buffer,
    save_speaker_state,
    get_speaker_state_path,
)
from core.processing import get_speaker_diarization
from rich.console import Console
from services.speaker_html_visualizer import SpeakerHTMLVisualizer
from jinja2 import Template
from fastapi import Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

console = Console()
router = APIRouter(prefix="/speakers", tags=["speakers"])

_html_visualizer = SpeakerHTMLVisualizer()


@router.get("")
async def get_speakers():
    """Get current speaker diarization information."""
    return get_speaker_diarization()


@router.get("/status")
def get_status() -> Dict:
    """Get current speakers status."""
    labeler = get_speaker_labeler()
    if not labeler:
        return {"status": "not_initialized"}
    health_status = labeler.get_health_status()
    return dict(health_status)


@router.get("/similarities")
def get_speaker_similarity_matrix() -> Dict:
    """Get current speakers similarity matrix."""
    labeler = get_speaker_labeler()
    if not labeler:
        return {"error": "Speaker labeler not initialized"}
    speaker_similarity_matrix = labeler.get_speaker_similarity_matrix()
    return dict(speaker_similarity_matrix)


@router.post("/consolidate")
async def consolidate_speakers_endpoint(
    threshold: float = Form(0.85),
    dry_run: bool = Form(False),
):
    """Consolidate similar speakers by merging those above similarity threshold.
    
    Parameters
    ----------
    threshold : float
        Similarity threshold above which speakers are merged (0.0 to 1.0).
    dry_run : bool
        If true, returns proposed merges without executing them.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    result = labeler.consolidate_speakers(threshold=threshold, dry_run=dry_run)
    if not dry_run:
        save_speaker_state()
    
    return {
        "success": True,
        **result,
    }


@router.post("/reset")
async def reset_speakers():
    """Reset speaker labeler state - fully clears all speaker tracking."""
    labeler = get_speaker_labeler()
    if labeler:
        labeler.reset()
    
    set_current_speaker(None)
    set_last_speaker_change_time(0.0)
    
    context_buffer = get_context_buffer()
    if context_buffer.segments:
        for segment_audio, metadata in context_buffer.segments:
            metadata["speaker_label"] = None
            metadata["speaker_confidence"] = 0.0
            metadata["speakers"] = []
    
    speaker_state_path = get_speaker_state_path()
    if speaker_state_path.exists():
        speaker_state_path.unlink()
    
    save_speaker_state()
    console.print("[warning]🔄 Speaker state fully reset: labeler + global state + context buffer[/warning]")
    
    return {"success": True, "message": "Speaker state reset"}


@router.post("/merge")
async def merge_speakers(label1: str = Form(...), label2: str = Form(...)):
    """Merge two speaker labels into one."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    result = labeler.merge_speakers(label1, label2)
    if result is None:
        raise HTTPException(status_code=400, detail=f"Could not merge {label1} and {label2}")
    
    save_speaker_state()
    return {"success": True, "merged_label": result}

# Visualizers

@router.get("/dashboard", response_class=HTMLResponse)
async def get_speaker_dashboard():
    """Get complete speaker diarization dashboard as HTML page."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    html_content = _html_visualizer.get_dashboard_html(labeler)
    return HTMLResponse(content=html_content)


@router.get("/plots", response_class=HTMLResponse)
async def get_speaker_plots(
    plots: Optional[str] = Query(None, description="Comma-separated plot names: pca,tsne,heatmap,timeline,dashboard")
):
    """Get speaker visualization plots as HTML page.
    
    Parameters
    ----------
    plots : str, optional
        Comma-separated list of plots to include.
        Options: pca, tsne, heatmap, timeline, dashboard
        Example: /speakers/plots?plots=pca,heatmap,timeline
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    # Parse plot selection
    plot_list = None
    if plots:
        plot_list = [p.strip() for p in plots.split(',')]
    
    html_content = _html_visualizer.get_plots_only_html(labeler, plots=plot_list)
    return HTMLResponse(content=html_content)


@router.get("/plot/{plot_name}")
async def get_single_plot(
    plot_name: str,
    format: str = Query("html", description="Response format: html or json (base64)")
):
    """Get a single speaker plot.
    
    Parameters
    ----------
    plot_name : str
        Plot name: pca, tsne, heatmap, timeline, dashboard
    format : str
        Response format: 'html' for full HTML page or 'json' for base64 string
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    # Generate single plot
    plot_methods = {
        'pca': lambda: _html_visualizer.visualizer.plot_centroids_2d(labeler, method='pca'),
        'tsne': lambda: _html_visualizer.visualizer.plot_centroids_2d(labeler, method='tsne'),
        'heatmap': lambda: _html_visualizer.visualizer.plot_similarity_heatmap(labeler),
        'timeline': lambda: _html_visualizer.visualizer.plot_timeline(labeler),
        'dashboard': lambda: _html_visualizer.visualizer.plot_dashboard(labeler),
    }
    
    if plot_name not in plot_methods:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown plot: {plot_name}. Options: {list(plot_methods.keys())}"
        )
    
    fig = plot_methods[plot_name]()
    
    if fig is None:
        raise HTTPException(status_code=400, detail="Not enough data to generate plot")
    
    img_base64 = _html_visualizer._figure_to_base64(fig)
    
    if format == "json":
        return JSONResponse(content={
            "plot": plot_name,
            "image_base64": img_base64
        })
    else:
        # Return as HTML page
        html_template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Speaker Plot - {{ plot_name }}</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 20px;
                    background: #f5f5f5;
                    text-align: center;
                }
                .plot-container {
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 20px auto;
                    max-width: 1200px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                img {
                    max-width: 100%;
                    height: auto;
                }
                .back-link {
                    display: inline-block;
                    margin: 20px;
                    padding: 10px 20px;
                    background: #667eea;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
                .back-link:hover {
                    background: #5a67d8;
                }
            </style>
        </head>
        <body>
            <a href="/speakers/dashboard" class="back-link">← Back to Dashboard</a>
            <div class="plot-container">
                <h2>{{ plot_name|title }} Plot</h2>
                <img src="data:image/png;base64,{{ image_base64 }}" alt="{{ plot_name }}">
            </div>
        </body>
        </html>
        """)
        
        return HTMLResponse(content=html_template.render(
            plot_name=plot_name,
            image_base64=img_base64
        ))


@router.get("/data/export")
async def export_speaker_data(
    format: str = Query("json", description="Export format: json or html")
):
    """Export complete speaker data and visualizations.
    
    Parameters
    ----------
    format : str
        'json' for raw data + base64 images, 'html' for complete dashboard
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    if format == "html":
        html_content = _html_visualizer.get_dashboard_html(labeler)
        return HTMLResponse(content=html_content)
    else:
        # Generate all plots as base64
        figures = _html_visualizer.generate_all_plots(labeler)
        
        # Collect all data
        export_data = {
            "speakers": labeler.get_all_speakers_info(),
            "health": labeler.get_health_status(),
            "similarities": labeler.get_speaker_similarity_matrix(),
            "plots": figures,
            "metadata": {
                "total_segments": labeler.total_segments_processed,
                "total_speakers_created": labeler.total_speakers_created,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return JSONResponse(content=export_data)
