"""
Speaker management routes with modern HTML dashboards.
"""
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
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
from config import TEMPLATES_DIR

console = Console()
router = APIRouter(prefix="/speakers", tags=["speakers"])

# Jinja2 template environment
_templates_dir = TEMPLATES_DIR / "speakers"
_templates_dir.mkdir(parents=True, exist_ok=True)

_jinja_env = Environment(
    loader=FileSystemLoader(str(_templates_dir)),
    autoescape=select_autoescape(['html', 'xml'])
)

console.print(f"[info]Speaker templates directory: {_templates_dir}[/info]")


def get_template(name: str):
    """Get a Jinja2 template by name with caching."""
    try:
        template = _jinja_env.get_template(name)
        console.print(f"[dim]Loaded template: {name}[/dim]")
        return template
    except Exception as e:
        console.print(f"[error]Failed to load template {name}: {e}[/error]")
        raise HTTPException(
            status_code=500, 
            detail=f"Template {name} not found or invalid"
        )


def render_template(name: str, context: dict = None) -> str:
    """Render a template with context."""
    template = get_template(name)
    return template.render(**(context or {}))


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
    
    console.print(f"[info]Consolidating speakers with threshold={threshold}, dry_run={dry_run}[/info]")
    result = labeler.consolidate_speakers(threshold=threshold, dry_run=dry_run)
    
    if not dry_run:
        save_speaker_state()
        console.print(f"[success]Speakers consolidated: {result.get('merges', [])}[/success]")
    
    return {
        "success": True,
        **result,
    }


@router.post("/reset")
async def reset_speakers():
    """Reset speaker labeler state - fully clears all speaker tracking."""
    labeler = get_speaker_labeler()
    if labeler:
        console.print("[warning]Resetting speaker labeler...[/warning]")
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
        console.print(f"[dim]Removed speaker state file: {speaker_state_path}[/dim]")
    
    save_speaker_state()
    console.print("[warning]🔄 Speaker state fully reset: labeler + global state + context buffer[/warning]")
    
    return {
        "success": True, 
        "message": "Speaker state reset successfully",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/merge")
async def merge_speakers(label1: str = Form(...), label2: str = Form(...)):
    """Merge two speaker labels into one."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    console.print(f"[info]Merging speakers: {label1} + {label2}[/info]")
    result = labeler.merge_speakers(label1, label2)
    
    if result is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Could not merge {label1} and {label2}"
        )
    
    save_speaker_state()
    console.print(f"[success]Merged into: {result}[/success]")
    
    return {
        "success": True, 
        "merged_label": result,
        "message": f"Successfully merged {label1} and {label2} into {result}"
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def get_speaker_dashboard(request: Request):
    """
    Get complete speaker diarization dashboard as modern HTML page.
    Uses Chart.js for interactive visualizations.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        console.print("[error]Speaker labeler not initialized for dashboard[/error]")
        raise HTTPException(
            status_code=400, 
            detail="Speaker labeler not initialized. Process some audio segments first."
        )
    
    console.print("[info]Rendering speaker dashboard...[/info]")
    
    try:
        # Try to load and render the dashboard template
        html_content = render_template("dashboard.html", {
            "title": "Speaker Diarization Dashboard",
            "timestamp": datetime.now().isoformat()
        })
        console.print("[success]Dashboard rendered successfully[/success]")
        return HTMLResponse(content=html_content)
    
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Failed to render dashboard: {e}[/error]")
        # Fallback to simple HTML
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Speaker Dashboard</title>
            <style>
                body {{ 
                    font-family: sans-serif; 
                    padding: 40px; 
                    text-align: center;
                    background: #0f172a;
                    color: #f1f5f9;
                }}
                .error {{ 
                    background: rgba(239,68,68,0.1); 
                    padding: 20px; 
                    border-radius: 10px;
                    max-width: 600px;
                    margin: 40px auto;
                }}
                .btn {{
                    display: inline-block;
                    padding: 10px 20px;
                    background: #3b82f6;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    margin: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>🎙️ Speaker Dashboard</h1>
            <div class="error">
                <p>Dashboard template not found or failed to render.</p>
                <p style="font-size:14px;color:#94a3b8;">Error: {str(e)}</p>
            </div>
            <a href="/speakers/plots" class="btn">📊 View Plots</a>
            <a href="/speakers/status" class="btn">📋 View Status</a>
        </body>
        </html>
        """)


@router.get("/plots", response_class=HTMLResponse)
async def get_speaker_plots(
    request: Request,
    plots: Optional[str] = Query(
        None, 
        description="Comma-separated plot names: pca,tsne,heatmap,timeline"
    )
):
    """
    Get speaker visualization plots as modern HTML page with Chart.js.
    
    Parameters
    ----------
    plots : str, optional
        Comma-separated list of plots to include.
        Options: pca, tsne, heatmap, timeline
        Example: /speakers/plots?plots=pca,heatmap,timeline
    """
    labeler = get_speaker_labeler()
    if not labeler:
        console.print("[error]Speaker labeler not initialized for plots[/error]")
        raise HTTPException(
            status_code=400, 
            detail="Speaker labeler not initialized. Process some audio segments first."
        )
    
    plot_list = None
    if plots:
        plot_list = [p.strip() for p in plots.split(',') if p.strip()]
        console.print(f"[info]Requested plots: {plot_list}[/info]")
    
    try:
        html_content = render_template("plots.html", {
            "title": "Speaker Visualization Plots",
            "requested_plots": plot_list,
            "timestamp": datetime.now().isoformat()
        })
        console.print("[success]Plots page rendered successfully[/success]")
        return HTMLResponse(content=html_content)
    
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Failed to render plots: {e}[/error]")
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Speaker Plots</title>
            <style>
                body {{ 
                    font-family: sans-serif; 
                    padding: 40px; 
                    text-align: center;
                    background: #0f172a;
                    color: #f1f5f9;
                }}
                .error {{ 
                    background: rgba(239,68,68,0.1); 
                    padding: 20px; 
                    border-radius: 10px;
                }}
            </style>
        </head>
        <body>
            <h1>📊 Speaker Plots</h1>
            <div class="error">
                <p>Failed to render plots page.</p>
                <p style="font-size:14px;color:#94a3b8;">Error: {str(e)}</p>
            </div>
        </body>
        </html>
        """)


@router.get("/plot/{plot_name}", response_class=HTMLResponse)
async def get_single_plot(
    request: Request,
    plot_name: str,
    format: str = Query("html", description="Response format: html or json (base64)")
):
    """
    Get a single speaker plot with interactive Chart.js visualization.
    
    Parameters
    ----------
    plot_name : str
        Plot name: pca, tsne, heatmap, timeline
    format : str
        Response format: 'html' for full HTML page or 'json' for base64 string
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(
            status_code=400, 
            detail="Speaker labeler not initialized"
        )
    
    valid_plots = {'pca', 'tsne', 'heatmap', 'timeline'}
    if plot_name not in valid_plots:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown plot: {plot_name}. Options: {list(valid_plots)}"
        )
    
    console.print(f"[info]Rendering single plot: {plot_name} (format={format})[/info]")
    
    if format == "json":
        # Return JSON with base64 image from matplotlib (backward compatibility)
        plot_methods = {
            'pca': lambda: labeler.plot_centroids_2d(method='pca') if hasattr(labeler, 'plot_centroids_2d') else None,
            'tsne': lambda: labeler.plot_centroids_2d(method='tsne') if hasattr(labeler, 'plot_centroids_2d') else None,
            'heatmap': lambda: labeler.plot_similarity_heatmap() if hasattr(labeler, 'plot_similarity_heatmap') else None,
            'timeline': lambda: labeler.plot_timeline() if hasattr(labeler, 'plot_timeline') else None,
        }
        
        try:
            fig = plot_methods[plot_name]()
            if fig is None:
                return JSONResponse(content={
                    "plot": plot_name,
                    "image_base64": None,
                    "message": "Not enough data to generate plot"
                })
            
            import io
            import base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            return JSONResponse(content={
                "plot": plot_name,
                "image_base64": img_base64
            })
        except Exception as e:
            console.print(f"[error]Failed to generate {plot_name} plot: {e}[/error]")
            return JSONResponse(content={
                "plot": plot_name,
                "image_base64": None,
                "error": str(e)
            })
    
    # Default: Return HTML page
    try:
        html_content = render_template("single_plot.html", {
            "plot_name": plot_name,
            "title": f"Speaker {plot_name.title()} Plot",
            "timestamp": datetime.now().isoformat()
        })
        console.print(f"[success]Single plot rendered: {plot_name}[/success]")
        return HTMLResponse(content=html_content)
    
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Failed to render single plot: {e}[/error]")
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{plot_name.title()} Plot</title>
            <style>
                body {{ 
                    font-family: sans-serif; 
                    padding: 40px; 
                    text-align: center;
                    background: #0f172a;
                    color: #f1f5f9;
                }}
                .btn {{
                    padding: 10px 20px;
                    background: #3b82f6;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <h1>📊 {plot_name.title()} Plot</h1>
            <p>Failed to render interactive plot.</p>
            <a href="/speakers/plots" class="btn">← Back to All Plots</a>
        </body>
        </html>
        """)


@router.get("/data/export")
async def export_speaker_data(
    format: str = Query("json", description="Export format: json or html")
):
    """
    Export complete speaker data and visualizations.
    
    Parameters
    ----------
    format : str
        'json' for raw data + base64 images, 'html' for complete dashboard
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(
            status_code=400, 
            detail="Speaker labeler not initialized"
        )
    
    console.print(f"[info]Exporting speaker data (format={format})[/info]")
    
    if format == "html":
        return await get_speaker_dashboard(Request)
    
    # JSON export
    import io
    import base64
    
    figures = {}
    
    # Generate matplotlib plots as base64 for backward compatibility
    try:
        if hasattr(labeler, 'plot_centroids_2d'):
            for method in ['pca', 'tsne']:
                try:
                    fig = labeler.plot_centroids_2d(method=method)
                    if fig:
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        figures[method] = {
                            "image_base64": base64.b64encode(buf.read()).decode('utf-8'),
                            "format": "png"
                        }
                except Exception as e:
                    console.print(f"[warning]Could not generate {method} plot: {e}[/warning]")
                    figures[method] = {"error": str(e)}
        
        if hasattr(labeler, 'plot_similarity_heatmap'):
            try:
                fig = labeler.plot_similarity_heatmap()
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    figures['heatmap'] = {
                        "image_base64": base64.b64encode(buf.read()).decode('utf-8'),
                        "format": "png"
                    }
            except Exception as e:
                console.print(f"[warning]Could not generate heatmap: {e}[/warning]")
                figures['heatmap'] = {"error": str(e)}
        
        if hasattr(labeler, 'plot_timeline'):
            try:
                fig = labeler.plot_timeline()
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    figures['timeline'] = {
                        "image_base64": base64.b64encode(buf.read()).decode('utf-8'),
                        "format": "png"
                    }
            except Exception as e:
                console.print(f"[warning]Could not generate timeline: {e}[/warning]")
                figures['timeline'] = {"error": str(e)}
    except Exception as e:
        console.print(f"[error]Plot generation failed: {e}[/error]")
    
    export_data = {
        "speakers": labeler.get_all_speakers_info() if hasattr(labeler, 'get_all_speakers_info') else [],
        "health": labeler.get_health_status() if hasattr(labeler, 'get_health_status') else {},
        "similarities": labeler.get_speaker_similarity_matrix() if hasattr(labeler, 'get_speaker_similarity_matrix') else {},
        "plots": figures,
        "metadata": {
            "total_segments": getattr(labeler, 'total_segments_processed', 0),
            "total_speakers_created": getattr(labeler, 'total_speakers_created', 0),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    console.print(f"[success]Exported data: {len(export_data['speakers'])} speakers, {len(figures)} plots[/success]")
    return JSONResponse(content=export_data)
