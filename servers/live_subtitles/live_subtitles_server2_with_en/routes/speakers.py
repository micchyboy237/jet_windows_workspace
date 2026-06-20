"""
Speaker management routes with modern HTML dashboards.
Aligned with SegmentSpeakerLabeler data structures.
"""
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console

from core.state import (
    get_speaker_labeler,
    set_current_speaker,
    set_last_speaker_change_time,
    get_context_buffer,
    save_speaker_state,
    get_speaker_state_path,
    get_last_n_segments_dir,
)
from core.processing import get_speaker_diarization
from services.audio_utils import get_audio_duration
from services.audio_config import SAMPLE_RATE
from services.config import TEMPLATES_DIR

console = Console()
router = APIRouter(prefix="/speakers", tags=["speakers"])

# ---------------------------------------------------------------------------
# Jinja2 template environment
# ---------------------------------------------------------------------------
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


def _check_component_exists(component_name: str) -> bool:
    """Check if a component template exists before including it."""
    component_path = _templates_dir / "components" / component_name
    exists = component_path.exists()
    if not exists:
        console.print(f"[warning]Component not found: {component_path}[/warning]")
    return exists


# ---------------------------------------------------------------------------
# JSON data endpoints
# ---------------------------------------------------------------------------

@router.get("")
async def get_speakers():
    """Get current speaker diarization information."""
    return get_speaker_diarization()


@router.get("/segments")
async def get_speaker_segments(
    label: Optional[str] = Query(None, description="Speaker label e.g. 'SPEAKER_01'. Omit for all speakers."),
):
    """Get segment info and raw embeddings for one or all speakers.

    Parameters
    ----------
    label : str, optional
        Speaker label (e.g. 'SPEAKER_01'). If omitted, returns all speakers.

    Returns
    -------
    Single SpeakerSegmentInfo dict if label is provided, else list of all.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")

    if label is not None:
        console.print(f"[info]get_segments: fetching segment info for label='{label}'[/info]")
        result = labeler.get_segments(label)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Speaker '{label}' not found")
        console.print(f"[success]get_segments: returned {result['embedding_count']} embeddings for {label}[/success]")
        return JSONResponse(content=result)

    console.print("[info]get_segments: fetching segment info for all speakers[/info]")
    results = labeler.get_segments()
    console.print(f"[success]get_segments: returned {len(results)} speakers[/success]")
    return JSONResponse(content=results)


@router.get("/status")
def get_status() -> Dict:
    """Get current speakers health status with full report."""
    labeler = get_speaker_labeler()
    if not labeler:
        return {"status": "not_initialized"}
    
    # Use the comprehensive health report if available
    if hasattr(labeler, 'get_speaker_health_report'):
        return labeler.get_speaker_health_report()
    
    # Fallback to basic health status
    return labeler.get_health_status()


@router.get("/centroids")
async def get_centroid_data():
    """Get speaker centroid coordinates and metadata for visualization.
    
    Returns centroid vectors, segment counts, quality scores,
    nearest neighbors, and per-dimension stats.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    # Use the comprehensive centroid stats method
    if hasattr(labeler, 'get_centroid_stats'):
        centroid_stats = labeler.get_centroid_stats()
        
        if "error" in centroid_stats:
            return {"centroids": {}, "total_speakers": 0, "embedding_dimension": 0, "error": centroid_stats["error"]}
        
        # Extract centroids in the format the frontend expects
        centroids = {}
        speakers_detail = centroid_stats.get("speakers", {})
        
        for label, detail in speakers_detail.items():
            centroids[label] = {
                "centroid_vector": detail.get("centroid_vector", []),
                "centroid_norm": detail.get("centroid_norm", 0),
                "top_dimensions": detail.get("top_dimensions", []),
                "segment_count": detail.get("segment_count", 0),
                "centroid_quality": detail.get("centroid_quality", 0),
                "first_seen": detail.get("first_seen", 0),
                "last_seen": detail.get("last_seen", 0),
                "active_duration": detail.get("active_duration", 0),
                "embedding_count": detail.get("embedding_count", 0),
                "avg_distance_to_others": detail.get("avg_distance_to_others", 0),
                "avg_similarity_to_others": detail.get("avg_similarity_to_others", 0),
                "nearest_neighbor": detail.get("nearest_neighbor"),
                "nearest_distance": detail.get("nearest_distance"),
                "nearest_similarity": detail.get("nearest_similarity"),
            }
        
        console.print(f"[info]Returning centroid data for {len(centroids)} speakers[/info]")
        
        return {
            "centroids": centroids,
            "total_speakers": centroid_stats.get("total_speakers", len(centroids)),
            "embedding_dimension": centroid_stats.get("embedding_dimension", 0),
            "similarity_matrix": centroid_stats.get("similarity_matrix", []),
            "distance_matrix": centroid_stats.get("distance_matrix", []),
        }
    
    # Fallback: Build from basic speaker info
    speakers_info = labeler.get_all_speakers_info()
    centroids = {}
    
    for label, info in speakers_info.items():
        coords = info.get("centroid_coordinates")
        if coords is not None:
            # Flatten if nested
            flat_coords = coords[0] if isinstance(coords[0], list) else coords
            centroids[label] = {
                "centroid_vector": flat_coords[:50] if len(flat_coords) > 50 else flat_coords,
                "centroid_norm": float(np.linalg.norm(flat_coords)),
                "top_dimensions": [],
                "segment_count": info.get("segment_count", 0),
                "centroid_quality": info.get("centroid_quality", 0),
                "first_seen": info.get("first_seen", 0),
                "last_seen": info.get("last_seen", 0),
                "active_duration": info.get("active_duration", 0),
                "embedding_count": 0,
            }
    
    console.print(f"[info]Returning centroid data for {len(centroids)} speakers (fallback)[/info]")
    
    return {
        "centroids": centroids,
        "total_speakers": len(centroids),
        "embedding_dimension": len(list(centroids.values())[0]["centroid_vector"]) if centroids else 0,
    }


@router.get("/centroid-distances")
async def get_centroid_distances():
    """Get pairwise distances between all speaker centroids."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    similarity_matrix = labeler.get_speaker_similarity_matrix()
    labels = similarity_matrix.get("labels", [])
    similarities = similarity_matrix.get("similarities", [])
    segment_counts = similarity_matrix.get("segment_counts", [])
    
    # Convert similarities to distances (1 - similarity = cosine distance)
    distances = []
    for i, row in enumerate(similarities):
        distance_row = []
        for j, sim in enumerate(row):
            distance_row.append(round(1.0 - sim, 4))
        distances.append(distance_row)
    
    return {
        "labels": labels,
        "distances": distances,
        "similarities": similarities,
        "segment_counts": segment_counts,
    }


@router.get("/centroid-comparison")
async def get_centroid_comparison(
    label1: str = Query(..., description="First speaker label"),
    label2: str = Query(..., description="Second speaker label"),
):
    """Compare two speaker centroids with detailed metrics.
    
    Returns cosine similarity/distance, euclidean distance, 
    per-dimension comparison, merge status, and speaker metadata.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    # Get speaker info
    speaker1 = labeler.get_speaker_info(label1)
    speaker2 = labeler.get_speaker_info(label2)
    
    if not speaker1 or not speaker2:
        missing = []
        if not speaker1:
            missing.append(label1)
        if not speaker2:
            missing.append(label2)
        raise HTTPException(
            status_code=404,
            detail=f"Speaker(s) not found: {', '.join(missing)}"
        )
    
    # Get raw centroid arrays
    centroids = labeler.get_centroid_arrays()
    centroid1 = centroids.get(label1)
    centroid2 = centroids.get(label2)
    
    if centroid1 is None or centroid2 is None:
        raise HTTPException(
            status_code=400,
            detail="One or both speakers have no valid centroid"
        )
    
    # Flatten centroids
    c1 = centroid1.flatten()
    c2 = centroid2.flatten()
    
    # Compute metrics
    from scipy.spatial.distance import cosine
    
    cos_dist = float(cosine(c1, c2))
    cos_sim = 1.0 - cos_dist
    euclidean_dist = float(np.linalg.norm(c1 - c2))
    
    # Per-dimension differences
    dim_diffs = np.abs(c1 - c2)
    top_different_dims = np.argsort(dim_diffs)[-10:][::-1]  # Top 10
    
    # Centroid norms
    norm1 = float(np.linalg.norm(c1))
    norm2 = float(np.linalg.norm(c2))
    
    # Get full centroid stats for additional metadata
    centroid_stats = labeler.get_centroid_stats() if hasattr(labeler, 'get_centroid_stats') else {}
    speakers_detail = centroid_stats.get("speakers", {})
    
    sp1_detail = speakers_detail.get(label1, {})
    sp2_detail = speakers_detail.get(label2, {})
    
    comparison = {
        "speaker1": {
            "label": label1,
            "segment_count": speaker1.get("segment_count", 0),
            "centroid_quality": speaker1.get("centroid_quality", 0),
            "centroid_norm": round(norm1, 4),
            "first_seen": speaker1.get("first_seen", 0),
            "last_seen": speaker1.get("last_seen", 0),
            "active_duration": speaker1.get("active_duration", 0),
            "nearest_neighbor": sp1_detail.get("nearest_neighbor"),
            "nearest_similarity": sp1_detail.get("nearest_similarity"),
        },
        "speaker2": {
            "label": label2,
            "segment_count": speaker2.get("segment_count", 0),
            "centroid_quality": speaker2.get("centroid_quality", 0),
            "centroid_norm": round(norm2, 4),
            "first_seen": speaker2.get("first_seen", 0),
            "last_seen": speaker2.get("last_seen", 0),
            "active_duration": speaker2.get("active_duration", 0),
            "nearest_neighbor": sp2_detail.get("nearest_neighbor"),
            "nearest_similarity": sp2_detail.get("nearest_similarity"),
        },
        "comparison": {
            "cosine_similarity": round(cos_sim, 4),
            "cosine_distance": round(cos_dist, 4),
            "euclidean_distance": round(euclidean_dist, 4),
            "would_merge": cos_sim >= labeler.consolidation_threshold,
            "merge_threshold": labeler.consolidation_threshold,
            "speaker1_centroid_vector": c1.tolist(),
            "speaker2_centroid_vector": c2.tolist(),
            "speaker1_norm": round(norm1, 4),
            "speaker2_norm": round(norm2, 4),
            "speaker1_segments": speaker1.get("segment_count", 0),
            "speaker2_segments": speaker2.get("segment_count", 0),
            "speaker1_quality": speaker1.get("centroid_quality", 0),
            "speaker2_quality": speaker2.get("centroid_quality", 0),
            "top_different_dimensions": [
                {
                    "dimension": int(d),
                    "value_speaker1": round(float(c1[d]), 6),
                    "value_speaker2": round(float(c2[d]), 6),
                    "diff": round(float(dim_diffs[d]), 6),
                    "abs_diff": round(float(dim_diffs[d]), 6),
                }
                for d in top_different_dims[:10]
            ],
            "dimension_count": len(c1),
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    console.print(
        f"[info]Centroid comparison: {label1} vs {label2} "
        f"(similarity={cos_sim:.3f}, distance={cos_dist:.3f})[/info]"
    )
    
    return comparison


@router.get("/similarities")
def get_speaker_similarity_matrix() -> Dict:
    """Get current speakers similarity matrix."""
    labeler = get_speaker_labeler()
    if not labeler:
        return {"error": "Speaker labeler not initialized"}
    return labeler.get_speaker_similarity_matrix()


@router.get("/centroid-quality-history")
async def get_centroid_quality_history():
    """Get centroid quality metrics, rejection history, and evolution data."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    # Get health stats
    health = labeler.get_centroid_health_stats() if hasattr(labeler, 'get_centroid_health_stats') else {}
    all_info = labeler.get_all_speakers_info()
    
    # Build quality timeline from update log
    quality_history = []
    if hasattr(labeler, '_centroid_update_log'):
        for entry in labeler._centroid_update_log[-50:]:
            quality_history.append({
                "label": entry.get("label"),
                "similarity": entry.get("similarity"),
                "match_type": entry.get("match_type"),
                "centroid_shift": entry.get("centroid_shift"),
                "segment_count": entry.get("segment_count"),
                "timestamp": entry.get("timestamp"),
            })
    
    return {
        "health_stats": health,
        "speakers": {
            label: {
                "quality": info.get("centroid_quality", 0),
                "segments": info.get("segment_count", 0),
                "centroid_shape": info.get("centroid_shape"),
            }
            for label, info in all_info.items()
        },
        "quality_history": quality_history,
        "total_rejected": health.get("total_updates_rejected", 0),
        "total_processed": health.get("total_segments_processed", 0),
        "rejection_rate": health.get("rejection_rate", 0),
    }


# ---------------------------------------------------------------------------
# HTML page endpoints
# ---------------------------------------------------------------------------

@router.get("/dashboard", response_class=HTMLResponse)
async def get_speaker_dashboard(request: Request):
    """Get complete speaker diarization dashboard as modern HTML page."""
    labeler = get_speaker_labeler()
    if not labeler:
        console.print("[error]Speaker labeler not initialized for dashboard[/error]")
        raise HTTPException(
            status_code=400,
            detail="Speaker labeler not initialized. Process some audio segments first."
        )
    
    console.print("[info]Rendering speaker dashboard...[/info]")
    
    try:
        html_content = render_template("dashboard.html", {
            "title": "Speaker Diarization Dashboard",
            "timestamp": datetime.now().isoformat(),
        })
        console.print("[success]Dashboard rendered successfully[/success]")
        return HTMLResponse(content=html_content)
    
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Failed to render dashboard: {e}[/error]")
        return HTMLResponse(
            content=_fallback_html(
                "Speaker Dashboard",
                str(e),
                [("🏠 Dashboard", "/speakers/dashboard"), ("📊 Plots", "/speakers/plots")],
            ),
            status_code=200,
        )


@router.get("/plots", response_class=HTMLResponse)
async def get_speaker_plots(
    request: Request,
    plots: Optional[str] = Query(
        None,
        description="Comma-separated plot names: pca,tsne,heatmap,timeline"
    ),
):
    """Get speaker visualization plots as modern HTML page with Chart.js."""
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
            "timestamp": datetime.now().isoformat(),
        })
        console.print("[success]Plots page rendered successfully[/success]")
        return HTMLResponse(content=html_content)
    
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Failed to render plots: {e}[/error]")
        return HTMLResponse(
            content=_fallback_html("Speaker Plots", str(e)),
            status_code=200,
        )


@router.get("/plot/{plot_name}", response_class=HTMLResponse)
async def get_single_plot(
    request: Request,
    plot_name: str,
    format: str = Query("html", description="Response format: html or json (base64)"),
):
    """Get a single speaker plot with interactive Chart.js visualization.
    
    Parameters
    ----------
    plot_name : str
        Plot name: pca, tsne, heatmap, timeline
    format : str
        Response format: 'html' for full HTML page or 'json' for base64 string
    
    Query params are forwarded to the frontend for tab/feature control:
        ?speaker=SPEAKER_01     -> highlight specific speaker
        ?speaker1=A&speaker2=B  -> open pairwise comparison
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    valid_plots = {'pca', 'tsne', 'heatmap', 'timeline'}
    if plot_name not in valid_plots:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown plot: {plot_name}. Options: {list(valid_plots)}"
        )
    
    console.print(f"[info]Rendering single plot: {plot_name} (format={format})[/info]")
    
    # Extract query params to forward to template
    query_params = dict(request.query_params)
    console.print(f"[dim]Query params: {query_params}[/dim]")
    
    if format == "json":
        # Return JSON with base64 image from matplotlib (backward compatibility)
        return await _get_plot_json(plot_name, labeler)
    
    # Build template context
    # Determine which components are available
    has_pairwise = _check_component_exists("pairwise_comparison.html")
    has_similarity_gauge = _check_component_exists("similarity_gauge.html")
    has_embedding_plot = _check_component_exists("speaker_embedding_plot.html")
    has_dimension_diff = _check_component_exists("dimension_diff_view.html")
    
    # Check if JS files exist
    js_dir = Path(__file__).parent.parent / "static" / "js" / "speakers"
    js_files = {
        "pairwise_comparison": (js_dir / "pairwise_comparison.js").exists(),
        "independent_analysis": (js_dir / "independent_analysis.js").exists(),
        "similarity_network": (js_dir / "similarity_network.js").exists(),
        "health_diagnostics": (js_dir / "health_diagnostics.js").exists(),
        "dimension_diff_view": (js_dir / "dimension_diff_view.js").exists(),
    }
    
    template_context = {
        "plot_name": plot_name,
        "title": f"Speaker {plot_name.title()} Plot",
        "timestamp": datetime.now().isoformat(),
        # Tab visibility flags
        "include_pairwise": has_pairwise,
        "include_speaker_analysis": True,
        "include_network": True,
        "include_health": True,
        # Component availability flags (NEW - these were computed but not passed!)
        "has_similarity_gauge": has_similarity_gauge,
        "has_embedding_plot": has_embedding_plot,
        "has_dimension_diff": has_dimension_diff,
        # JS availability flags
        "has_pairwise_js": js_files["pairwise_comparison"],
        "has_independent_js": js_files["independent_analysis"],
        "has_similarity_js": js_files["similarity_network"],
        "has_health_js": js_files["health_diagnostics"],
        "has_dimension_diff_js": js_files["dimension_diff_view"],
        # Query params forwarded for frontend
        "query_speaker": query_params.get("speaker", ""),
        "query_speaker1": query_params.get("speaker1", ""),
        "query_speaker2": query_params.get("speaker2", ""),
        "query_view": query_params.get("view", "scatter"),
    }
    
    try:
        html_content = render_template("single_plot.html", template_context)
        console.print(f"[success]Single plot rendered: {plot_name}[/success]")
        return HTMLResponse(content=html_content)
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        console.print(f"[error]Failed to render single plot: {error_msg}[/error]")
        
        # Check for common causes
        if "maximum recursion" in error_msg.lower():
            console.print("[warning]Recursion detected - likely missing component file[/warning]")
            console.print(f"[info]Components dir: {_templates_dir / 'components'}[/info]")
            if (_templates_dir / "components").exists():
                console.print(f"[info]Available: {list((_templates_dir / 'components').glob('*.html'))}[/info]")
        
        return HTMLResponse(
            content=_fallback_html(
                f"{plot_name.title()} Plot",
                error_msg,
                [
                    ("🏠 Dashboard", "/speakers/dashboard"),
                    ("📊 All Plots", "/speakers/plots"),
                ],
            ),
            status_code=200,
        )


async def _get_plot_json(plot_name: str, labeler) -> JSONResponse:
    """Generate plot as base64 JSON (backward compatibility)."""
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
                "message": "Not enough data to generate plot",
            })
        
        import io
        import base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return JSONResponse(content={
            "plot": plot_name,
            "image_base64": img_base64,
        })
    except Exception as e:
        console.print(f"[error]Failed to generate {plot_name} plot: {e}[/error]")
        return JSONResponse(content={
            "plot": plot_name,
            "image_base64": None,
            "error": str(e),
        })


# ---------------------------------------------------------------------------
# Action endpoints
# ---------------------------------------------------------------------------

@router.post("/consolidate")
async def consolidate_speakers_endpoint(
    threshold: float = Form(0.85),
    dry_run: bool = Form(False),
):
    """Consolidate similar speakers by merging those above similarity threshold."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    console.print(f"[info]Consolidating speakers: threshold={threshold}, dry_run={dry_run}[/info]")
    result = labeler.consolidate_speakers(threshold=threshold, dry_run=dry_run)
    
    if not dry_run:
        save_speaker_state()
        console.print(f"[success]Speakers consolidated: {result.get('merges_performed', [])}[/success]")
    
    return {"success": True, **result}


@router.post("/reset")
async def reset_speakers():
    """Reset speaker labeler state - fully clears all speaker tracking."""
    labeler = get_speaker_labeler()
    if labeler:
        console.print("[warning]Resetting speaker labeler...[/warning]")
        labeler.reset()
    
    set_current_speaker(None)
    set_last_speaker_change_time(0.0)
    
    # Clear context buffer speaker labels
    context_buffer = get_context_buffer()
    if context_buffer and context_buffer.segments:
        for segment_audio, metadata in context_buffer.segments:
            metadata["speaker_label"] = None
            metadata["speaker_confidence"] = 0.0
            metadata["speakers"] = []
    
    # Remove persisted state
    speaker_state_path = get_speaker_state_path()
    if speaker_state_path and speaker_state_path.exists():
        speaker_state_path.unlink()
        console.print(f"[dim]Removed speaker state file: {speaker_state_path}[/dim]")
    
    save_speaker_state()
    console.print("[warning]🔄 Speaker state fully reset[/warning]")
    
    return {
        "success": True,
        "message": "Speaker state reset successfully",
        "timestamp": datetime.now().isoformat(),
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
            detail=f"Could not merge {label1} and {label2}",
        )
    
    save_speaker_state()
    console.print(f"[success]Merged into: {result}[/success]")
    
    return {
        "success": True,
        "merged_label": result,
        "message": f"Successfully merged {label1} and {label2} into {result}",
    }


@router.post("/rename")
async def rename_speaker(
    old_label: str = Form(...),
    new_label: str = Form(...),
):
    """Rename a speaker label, or merge if target already exists."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    if old_label not in labeler._speakers:
        raise HTTPException(status_code=404, detail=f"Speaker {old_label} not found")
    
    if new_label in labeler._speakers:
        # Target exists → merge instead
        console.print(f"[info]Renaming {old_label} → {new_label} (target exists, merging)[/info]")
        result = labeler.merge_speakers(new_label, old_label)
        save_speaker_state()
        return {
            "success": True,
            "action": "merge",
            "merged_label": result,
            "message": f"Merged {old_label} into {new_label}",
        }
    
    console.print(f"[info]Renaming speaker: {old_label} → {new_label}[/info]")
    
    ref = labeler._speakers[old_label]
    ref.label = new_label
    labeler._speakers[new_label] = ref
    del labeler._speakers[old_label]
    
    # Update creation times
    if hasattr(labeler, '_speaker_creation_times') and old_label in labeler._speaker_creation_times:
        labeler._speaker_creation_times[new_label] = labeler._speaker_creation_times.pop(old_label)
    
    # Update label history
    labeler._label_history = [
        (t, new_label if l == old_label else l)
        for t, l in labeler._label_history
    ]
    
    # Update merge history references
    if hasattr(labeler, '_merge_history'):
        for entry in labeler._merge_history:
            if entry.get("source") == old_label:
                entry["source"] = new_label
            if entry.get("target") == old_label:
                entry["target"] = new_label
    
    save_speaker_state()
    console.print(f"[success]Speaker renamed: {old_label} → {new_label}[/success]")
    
    return {
        "success": True,
        "action": "rename",
        "old_label": old_label,
        "new_label": new_label,
        "message": f"Renamed {old_label} to {new_label}",
    }


@router.get("/data/export")
async def export_speaker_data(
    format: str = Query("json", description="Export format: json or html"),
):
    """Export complete speaker data and visualizations."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    console.print(f"[info]Exporting speaker data (format={format})[/info]")
    
    if format == "html":
        return await get_speaker_dashboard(Request)
    
    # JSON export with all available data
    export_data = {
        "speakers": labeler.get_all_speakers_info() if hasattr(labeler, 'get_all_speakers_info') else {},
        "health": labeler.get_speaker_health_report() if hasattr(labeler, 'get_speaker_health_report') else labeler.get_health_status(),
        "similarities": labeler.get_speaker_similarity_matrix(),
        "centroid_stats": labeler.get_centroid_stats() if hasattr(labeler, 'get_centroid_stats') else {},
        "merge_history": labeler.get_merge_history() if hasattr(labeler, 'get_merge_history') else [],
        "metadata": {
            "total_segments": getattr(labeler, 'total_segments_processed', 0),
            "total_speakers_created": getattr(labeler, 'total_speakers_created', 0),
            "timestamp": datetime.now().isoformat(),
        },
    }
    
    console.print(f"[success]Exported data: {len(export_data['speakers'])} speakers[/success]")
    return JSONResponse(content=export_data)


@router.get("/segment/{segment_id}/audio")
async def get_segment_audio(segment_id: str, request: Request):
    """
    Get the audio data for a specific segment (for playback/download).
    Supports HTTP Range requests for proper streaming/seeking.
    
    CRITICAL: Always returns 206 Partial Content, even on first request.
    This ensures the browser knows the total file size and can calculate
    byte offsets for seeking.
    """
    from fastapi.responses import Response, StreamingResponse
    import os
    
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    console.print(f"[info]Fetching audio for segment: {segment_id}[/]")
    
    audio_path = None
    audio_source = "unknown"
    
    # === APPROACH 1: Check permanent segment audio directory ===
    try:
        from services.config import SEGMENT_AUDIO_DIR
        if SEGMENT_AUDIO_DIR and SEGMENT_AUDIO_DIR.exists():
            candidate = SEGMENT_AUDIO_DIR / f"{segment_id}.wav"
            if candidate.exists():
                audio_path = candidate
                audio_source = "permanent_storage"
                console.print(f"[success]Audio found in permanent storage: {audio_path}[/]")
    except Exception as e:
        console.print(f"[warning]Error checking permanent storage: {e}[/]")
    
    # === APPROACH 2: Check context buffer (fallback) ===
    if not audio_path:
        try:
            context_buffer = get_context_buffer()
            if context_buffer and hasattr(context_buffer, 'segments'):
                for segment_audio, metadata in context_buffer.segments:
                    meta_segment_id = metadata.get('segment_id', '')
                    if meta_segment_id == segment_id:
                        import io
                        import wave
                        import tempfile
                        
                        # Convert audio to numpy array
                        if isinstance(segment_audio, torch.Tensor):
                            audio_np = segment_audio.detach().cpu().numpy()
                        elif isinstance(segment_audio, np.ndarray):
                            audio_np = segment_audio
                        else:
                            audio_np = np.array(segment_audio, dtype=np.float32)
                        
                        # Ensure 1D array
                        if audio_np.ndim > 1:
                            audio_np = audio_np.flatten()
                        
                        # Convert float [-1,1] to int16 PCM
                        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
                        
                        # Write to temp file for streaming
                        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        with wave.open(tmp.name, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(audio_int16.tobytes())
                        
                        audio_path = Path(tmp.name)
                        audio_source = "context_buffer"
                        console.print(f"[success]Audio created from context buffer: {len(audio_int16)} samples[/]")
                        break
        except Exception as e:
            console.print(f"[warning]Error creating audio from context buffer: {e}[/]")
    
    # === APPROACH 3: Check disk fallback ===
    if not audio_path:
        try:
            last_n_dir = get_last_n_segments_dir()
            if last_n_dir and last_n_dir.exists():
                candidate = last_n_dir / f"{segment_id}.wav"
                if candidate.exists():
                    audio_path = candidate
                    audio_source = "disk"
                    console.print(f"[success]Audio found on disk: {audio_path}[/]")
                else:
                    # Try partial match
                    for f in last_n_dir.glob("*.wav"):
                        if segment_id in f.name:
                            audio_path = f
                            audio_source = "disk_partial_match"
                            console.print(f"[success]Audio found on disk (partial): {audio_path}[/]")
                            break
        except Exception as e:
            console.print(f"[warning]Error checking disk: {e}[/]")
    
    if not audio_path or not audio_path.exists():
        console.print(f"[error]No audio found for segment: {segment_id}[/]")
        raise HTTPException(status_code=404, detail="No audio data found")
    
    # ═══════════════════════════════════════════════════════════
    # ✅ CRITICAL FIX: Always serve with Range support
    # ═══════════════════════════════════════════════════════════
    file_size = audio_path.stat().st_size
    duration_sec = get_audio_duration(str(audio_path))
    
    # ✅ Define initial buffer size (like demo1's INITIAL_BUFFER)
    INITIAL_BUFFER = 512 * 1024  # 512 KB initial chunk
    
    # Parse Range header (or use default initial chunk)
    range_header = request.headers.get("range")
    
    if range_header:
        # Handle explicit Range request from browser
        try:
            range_value = range_header.replace("bytes=", "")
            start_str, end_str = range_value.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            
            if start >= file_size:
                raise HTTPException(status_code=416, detail="Range not satisfiable")
            
            end = min(end, file_size - 1)
            chunk_size = end - start + 1
            
            console.print(
                f"[info]Serving audio range: bytes {start}-{end}/{file_size} "
                f"({chunk_size} bytes)[/]"
            )
        except (ValueError, IndexError) as e:
            console.print(f"[warning]Invalid Range header: {range_header} - {e}[/]")
            # Fall through to default initial chunk
            start = 0
            end = min(INITIAL_BUFFER - 1, file_size - 1)
            chunk_size = end - start + 1
    else:
        # ✅ CRITICAL: No Range header → serve initial chunk as 206
        # This is the same behavior as demo1's parse_range_header
        # The browser needs this to know the total file size
        start = 0
        end = min(INITIAL_BUFFER - 1, file_size - 1)
        chunk_size = end - start + 1
        console.print(
            f"[info]No Range header → serving initial chunk: "
            f"bytes {start}-{end}/{file_size} ({chunk_size} bytes)[/]"
        )
    
    # ✅ ALWAYS return 206 Partial Content
    def file_iterator(file_path, start_byte, end_byte):
        with open(file_path, "rb") as f:
            f.seek(start_byte)
            remaining = end_byte - start_byte + 1
            while remaining > 0:
                read_size = min(8192, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data
    
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "audio/wav",
        "Content-Disposition": f"inline; filename={segment_id}.wav",
        "X-Segment-ID": segment_id,
        "X-Audio-Source": audio_source,
        "X-Audio-Duration": str(round(duration_sec, 3)),
        "Cache-Control": "public, max-age=3600",
        "X-Accel-Buffering": "no",  # ✅ Like demo1 - disable proxy buffering
    }
    
    console.print(
        f"[success]Serving 206: bytes {start}-{end}/{file_size} "
        f"({chunk_size} bytes, source={audio_source})[/]"
    )
    
    return StreamingResponse(
        file_iterator(audio_path, start, end),
        status_code=206,  # ✅ ALWAYS 206
        headers=headers,
        media_type="audio/wav",
    )


@router.head("/segment/{segment_id}/audio")
async def get_segment_audio_head(segment_id: str, request: Request):
    """
    HEAD request returns headers only (duration, size, type).
    Allows browser to probe audio availability without downloading the file.
    
    The browser may send HEAD before GET to check:
    - Whether the resource exists (404 check)
    - Content-Type for proper handling
    - Content-Length for download progress calculation
    - Accept-Ranges to know if seeking is supported
    """
    from fastapi.responses import Response
    
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    console.print(f"[info]HEAD request for segment audio: {segment_id}[/]")
    
    audio_path = None
    audio_source = "unknown"
    
    # === APPROACH 1: Check permanent segment audio directory ===
    try:
        from services.config import SEGMENT_AUDIO_DIR
        if SEGMENT_AUDIO_DIR and SEGMENT_AUDIO_DIR.exists():
            candidate = SEGMENT_AUDIO_DIR / f"{segment_id}.wav"
            if candidate.exists():
                audio_path = candidate
                audio_source = "permanent_storage"
                console.print(f"[dim]HEAD: Audio found in permanent storage: {audio_path}[/]")
    except Exception as e:
        console.print(f"[dim]HEAD: Error checking permanent storage: {e}[/]")
    
    # === APPROACH 2: Check context buffer (fallback) ===
    if not audio_path:
        try:
            context_buffer = get_context_buffer()
            if context_buffer and hasattr(context_buffer, 'segments'):
                for segment_audio, metadata in context_buffer.segments:
                    meta_segment_id = metadata.get('segment_id', '')
                    if meta_segment_id == segment_id:
                        import wave
                        import tempfile
                        
                        # Convert audio to numpy array
                        if isinstance(segment_audio, torch.Tensor):
                            audio_np = segment_audio.detach().cpu().numpy()
                        elif isinstance(segment_audio, np.ndarray):
                            audio_np = segment_audio
                        else:
                            audio_np = np.array(segment_audio, dtype=np.float32)
                        
                        # Ensure 1D array
                        if audio_np.ndim > 1:
                            audio_np = audio_np.flatten()
                        
                        # Convert float [-1,1] to int16 PCM
                        audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
                        
                        # Write to temp file
                        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        with wave.open(tmp.name, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(audio_int16.tobytes())
                        
                        audio_path = Path(tmp.name)
                        audio_source = "context_buffer"
                        console.print(f"[dim]HEAD: Audio created from context buffer: {len(audio_int16)} samples[/]")
                        break
        except Exception as e:
            console.print(f"[dim]HEAD: Error creating audio from context buffer: {e}[/]")
    
    # === APPROACH 3: Check disk fallback ===
    if not audio_path:
        try:
            last_n_dir = get_last_n_segments_dir()
            if last_n_dir and last_n_dir.exists():
                candidate = last_n_dir / f"{segment_id}.wav"
                if candidate.exists():
                    audio_path = candidate
                    audio_source = "disk"
                    console.print(f"[dim]HEAD: Audio found on disk: {audio_path}[/]")
                else:
                    # Try partial match
                    for f in last_n_dir.glob("*.wav"):
                        if segment_id in f.name:
                            audio_path = f
                            audio_source = "disk_partial_match"
                            console.print(f"[dim]HEAD: Audio found on disk (partial): {audio_path}[/]")
                            break
        except Exception as e:
            console.print(f"[dim]HEAD: Error checking disk: {e}[/]")
    
    if not audio_path or not audio_path.exists():
        console.print(f"[warning]HEAD: No audio found for segment: {segment_id}[/]")
        raise HTTPException(status_code=404, detail="No audio data found")
    
    # === Serve HEAD response (headers only, no body) ===
    file_size = audio_path.stat().st_size
    duration_sec = get_audio_duration(str(audio_path))
    
    console.print(
        f"[dim]HEAD: Returning metadata for {segment_id}: "
        f"size={file_size} bytes, duration={duration_sec:.3f}s, "
        f"source={audio_source}[/]"
    )
    
    headers = {
        "Content-Length": str(file_size),
        "Accept-Ranges": "bytes",
        "Content-Type": "audio/wav",
        "Content-Disposition": f"inline; filename={segment_id}.wav",
        "X-Segment-ID": segment_id,
        "X-Audio-Source": audio_source,
        "X-Audio-Duration": str(round(duration_sec, 3)),
        "Cache-Control": "public, max-age=3600",
    }
    
    # HEAD response: 200 OK with headers, no body
    return Response(
        status_code=200,
        headers=headers,
    )


@router.get("/segment/{segment_id}", response_class=HTMLResponse)
async def get_segment_detail_page(request: Request, segment_id: str):
    """
    Serve a detailed page for a specific segment with play/download audio buttons.
    Uses reusable audio_player.html template with Howler.js for robust streaming.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(
            status_code=400,
            detail="Speaker labeler not initialized. Process some audio segments first."
        )
    
    console.print(f"[info]Rendering segment detail page for: {segment_id}[/]")
    
    if not hasattr(labeler, 'get_segment_detail'):
        raise HTTPException(
            status_code=500,
            detail="Segment detail not available. Update speaker_metrics_mixin."
        )
    
    segment_info = labeler.get_segment_detail(segment_id)
    if segment_info is None:
        console.print(f"[warning]Segment not found: {segment_id}[/]")
        try:
            html_content = render_template("segment_detail.html", {
                "title": f"Segment: {segment_id}",
                "segment_id": segment_id,
                "found": False,
                "timestamp": datetime.now().isoformat(),
                "has_audio": False,
                "audio_api_base": "/segments",
            })
            return HTMLResponse(content=html_content, status_code=404)
        except Exception:
            return HTMLResponse(
                content=_fallback_html(
                    f"Segment: {segment_id}",
                    f"Segment '{segment_id}' not found in any speaker's data.",
                    [("📊 Metrics", "/speakers/metrics"), ("🏠 Dashboard", "/speakers/dashboard")],
                ),
                status_code=404,
            )
    
    # Check audio availability (permanent storage first, then context buffer, then disk)
    has_audio = False
    audio_source = ""
    audio_duration = segment_info.get("segment_duration", 0.0)
    audio_sample_rate = SAMPLE_RATE
    
    # === Check permanent storage ===
    try:
        from services.config import SEGMENT_AUDIO_DIR
        if SEGMENT_AUDIO_DIR and SEGMENT_AUDIO_DIR.exists():
            audio_path = SEGMENT_AUDIO_DIR / f"{segment_id}.wav"
            if audio_path.exists():
                has_audio = True
                audio_source = "permanent_storage"
                # ✅ Use utility for file duration
                disk_duration = get_audio_duration(str(audio_path))
                if audio_duration <= 0.0 or disk_duration > 0:
                    audio_duration = disk_duration
                console.print(f"[dim]Audio found in permanent storage: {audio_path} ({disk_duration:.3f}s)[/]")
    except Exception as e:
        console.print(f"[dim]Error checking permanent storage: {e}[/]")
    
    # === Check context buffer ===
    if not has_audio:
        try:
            context_buffer = get_context_buffer()
            if context_buffer and hasattr(context_buffer, 'segments'):
                for segment_audio, metadata in context_buffer.segments:
                    if metadata.get('segment_id') == segment_id:
                        has_audio = True
                        audio_source = "context_buffer"
                        # ✅ Use utility for duration
                        raw_duration = get_audio_duration(segment_audio, sr=SAMPLE_RATE) if segment_audio is not None else 0.0
                        if audio_duration <= 0.0 and raw_duration > 0.0:
                            audio_duration = raw_duration
                        break
        except Exception as e:
            console.print(f"[dim]Could not check audio availability: {e}[/]")
    
    # === Check disk fallback ===
    if not has_audio:
        try:
            last_n_dir = get_last_n_segments_dir()
            if last_n_dir and last_n_dir.exists():
                audio_path = last_n_dir / f"{segment_id}.wav"
                if audio_path.exists():
                    has_audio = True
                    audio_source = "disk"
                    # ✅ Use utility for file duration
                    disk_duration = get_audio_duration(str(audio_path))
                    if audio_duration <= 0.0:
                        audio_duration = disk_duration
        except Exception:
            pass

    console.print(
        f"[info]Segment detail: speaker={segment_info['speaker_label']}, "
        f"timestamp={segment_info['timestamp']:.2f}s, "
        f"duration={segment_info['segment_duration']:.3f}s, "
        f"audio={'yes' if has_audio else 'no'} ({audio_source}), "
        f"audio_duration={audio_duration:.3f}s[/]"
    )
    
    try:
        html_content = render_template("segment_detail.html", {
            "title": f"Segment: {segment_id}",
            "segment_id": segment_id,
            "found": True,
            "speaker_label": segment_info["speaker_label"],
            "timestamp": datetime.now().isoformat(),
            "segment_timestamp": segment_info["timestamp"],
            "segment_duration": segment_info["segment_duration"],  # Metadata duration (speech-filtered)
            "embedding_index": segment_info["embedding_index"],
            "embedding_dim": segment_info["embedding_dim"],
            "speaker_segment_count": segment_info["speaker_segment_count"],
            "speaker_first_seen": segment_info["speaker_first_seen"],
            "speaker_last_seen": segment_info["speaker_last_seen"],
            "speaker_active_duration": segment_info["speaker_active_duration"],
            "centroid_quality": segment_info["centroid_quality"],
            "has_audio": has_audio,
            "audio_source": audio_source,
            "audio_sample_rate": audio_sample_rate,
            "audio_duration": audio_duration,  # Raw waveform duration for playback info
            # ✅ NEW: Pass API base URL so the reusable audio_player.html knows where to fetch audio
            "audio_api_base": "/segments",
        })
        console.print(f"[success]Segment detail page rendered for {segment_id}[/]")
        return HTMLResponse(content=html_content)
    except Exception as e:
        console.print(f"[error]Failed to render segment detail: {e}[/]")
        return HTMLResponse(
            content=_fallback_html(
                f"Segment: {segment_id}",
                str(e),
                [("📊 Metrics", "/speakers/metrics"), ("🏠 Dashboard", "/speakers/dashboard")],
            ),
            status_code=200,
        )


# ============================================================
# Speaker Metrics Endpoints (uses SpeakerMetricsMixin methods)
# ============================================================

@router.get("/metrics/data")
async def get_speaker_metrics_data(
    label: Optional[str] = Query(
        None,
        description="Filter intra-speaker metrics to a specific speaker label (e.g., 'SPEAKER_01')",
    ),
):
    """
    Get comprehensive speaker metrics data for the frontend dashboard.
    
    Returns intra-speaker variance (per speaker) and inter-speaker separation
    (pairwise between centroids). Used by speaker_metrics.html.
    
    Parameters
    ----------
    label : str, optional
        Specific speaker label to filter intra-speaker results.
    
    Returns
    -------
    JSON with structure:
    {
        "intra_speaker": {
            "speakers": [{label, segmentsCount, health, meanDist, stdDev, 
                          minDist, maxDist, segments: [{id, d}]}],
            "overall_status": "healthy|warning|unhealthy",
            "total_speakers_analyzed": int
        },
        "inter_speaker": {
            "meanSeparation", "stdSeparation", "minSeparation", "maxSeparation",
            "health": "healthy|warning|unhealthy",
            "pairwise": [{from, to, distance}],
            "num_speakers": int
        },
        "timestamp": "ISO datetime"
    }
    """
    labeler = get_speaker_labeler()
    if not labeler:
        console.print("[error]Speaker labeler not initialized for metrics[/]")
        raise HTTPException(
            status_code=400,
            detail="Speaker labeler not initialized. Process some audio segments first."
        )

    # Check if the labeler has the mixin methods
    if not hasattr(labeler, 'get_speaker_metrics'):
        console.print("[error]SpeakerMetricsMixin not applied to labeler[/]")
        raise HTTPException(
            status_code=500,
            detail="Speaker metrics not available. Mixin not applied."
        )

    console.print(f"[info]Fetching speaker metrics data (label={label or 'all'})[/]")
    try:
        metrics_data = labeler.get_speaker_metrics(label=label)
        console.print(
            f"[success]Speaker metrics: {metrics_data['intra_speaker']['total_speakers_analyzed']} "
            f"intra, {metrics_data['inter_speaker']['num_speakers']} inter speakers[/]"
        )
        return JSONResponse(content=metrics_data)
    except Exception as e:
        console.print(f"[error]Failed to compute speaker metrics: {e}[/]")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute speaker metrics: {str(e)}"
        )


@router.get("/metrics/intra")
async def get_intra_speaker_metrics(
    label: Optional[str] = Query(None, description="Filter to specific speaker label"),
):
    """Get intra-speaker variance metrics only."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    if not hasattr(labeler, 'compute_intra_speaker_metrics'):
        raise HTTPException(status_code=500, detail="Speaker metrics mixin not applied")
    
    console.print(f"[info]Fetching intra-speaker metrics (label={label or 'all'})[/]")
    result = labeler.compute_intra_speaker_metrics(label=label)
    return JSONResponse(content=result)


@router.get("/metrics/inter")
async def get_inter_speaker_metrics():
    """Get inter-speaker separation metrics only."""
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    if not hasattr(labeler, 'compute_inter_speaker_metrics'):
        raise HTTPException(status_code=500, detail="Speaker metrics mixin not applied")
    
    console.print("[info]Fetching inter-speaker metrics[/]")
    result = labeler.compute_inter_speaker_metrics()
    return JSONResponse(content=result)


@router.get("/metrics", response_class=HTMLResponse)
async def get_speaker_metrics_page(request: Request):
    """
    Serve the speaker metrics HTML dashboard page.
    
    This returns the speaker_metrics.html template as a complete page.
    The frontend JavaScript will then fetch data from /speakers/metrics/data.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        console.print("[warning]Speaker labeler not initialized for metrics page[/]")
        # Still serve the page - it will show empty/error state

    console.print("[info]Serving speaker metrics HTML page[/]")
    try:
        html_content = render_template("speaker_metrics.html", {
            "title": "Speaker Metrics Dashboard",
            "timestamp": datetime.now().isoformat(),
        })
        console.print("[success]Speaker metrics page rendered[/]")
        return HTMLResponse(content=html_content)
    except Exception as e:
        console.print(f"[error]Failed to render speaker metrics page: {e}[/]")
        # Try serving the raw file if template rendering fails
        metrics_html_path = _templates_dir / "speaker_metrics.html"
        if metrics_html_path.exists():
            console.print("[info]Falling back to raw HTML file[/]")
            return HTMLResponse(content=metrics_html_path.read_text(encoding='utf-8'))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render metrics page: {str(e)}"
        )


@router.get("/metrics/health")
async def get_speaker_metrics_health():
    """
    Quick health check endpoint for speaker metrics.
    Returns a simple summary of overall speaker health.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    if not hasattr(labeler, 'get_speaker_metrics'):
        raise HTTPException(status_code=500, detail="Speaker metrics mixin not applied")
    
    metrics = labeler.get_speaker_metrics()
    return JSONResponse(content={
        "intra_overall_status": metrics["intra_speaker"]["overall_status"],
        "inter_health": metrics["inter_speaker"]["health"],
        "total_speakers": metrics["intra_speaker"]["total_speakers_analyzed"],
        "mean_separation": metrics["inter_speaker"]["meanSeparation"],
        "timestamp": metrics["timestamp"],
    })


# ---------------------------------------------------------------------------
# Fallback HTML helper
# ---------------------------------------------------------------------------

def _fallback_html(
    title: str,
    error: str = "",
    links: List[tuple] = None,
) -> str:
    """Generate a simple fallback HTML page when templates fail."""
    if links is None:
        links = [("🏠 Dashboard", "/speakers/dashboard")]
    
    links_html = "".join(
        f'<a href="{url}" class="btn">{label}</a>'
        for label, url in links
    )
    
    error_html = ""
    if error:
        error_html = f"""
        <div class="error">
            <p><strong>Render Error</strong></p>
            <p style="font-size:14px;color:#94a3b8;">{error[:300]}</p>
        </div>"""
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                padding: 40px;
                text-align: center;
                background: #0f172a;
                color: #f1f5f9;
            }}
            .error {{
                background: rgba(239,68,68,0.1);
                border: 1px solid rgba(239,68,68,0.3);
                padding: 20px;
                border-radius: 10px;
                max-width: 600px;
                margin: 40px auto;
            }}
            .info {{
                background: rgba(59,130,246,0.1);
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                font-size: 14px;
                color: #94a3b8;
                max-width: 600px;
                margin: 20px auto;
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
            .btn:hover {{ background: #2563eb; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        {error_html}
        <div class="info">
            <p>⚠️ The template may include components that are missing or failed to compile.</p>
            <p>Check <code>templates/speakers/components/</code> and <code>static/js/speakers/</code></p>
        </div>
        <div>{links_html}</div>
    </body>
    </html>
    """
