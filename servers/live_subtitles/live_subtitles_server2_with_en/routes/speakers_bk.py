"""
Speaker management routes with modern HTML dashboards.
Aligned with SegmentSpeakerLabeler data structures.
"""
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
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
)
from core.processing import get_speaker_diarization
from config import TEMPLATES_DIR

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


# ---------------------------------------------------------------------------
# Segment single / multi speaker endpoints
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: compute per-embedding details for a single speaker
# ---------------------------------------------------------------------------

def _compute_per_embedding_details(labeler, label: str) -> List[Dict]:
    """Compute per-embedding cosine similarity to centroid and outlier status.
    
    Returns a list of per-embedding metrics useful for sparklines and 
    scatter plots in the UI.
    
    Parameters
    ----------
    labeler : SegmentSpeakerLabeler
        The speaker labeler instance.
    label : str
        Speaker label (e.g. 'SPEAKER_01').
    
    Returns
    -------
    List[Dict] with keys: index, cosine_sim_to_centroid, is_outlier, 
                          cosine_sim_to_nearest_other
    """
    if label not in labeler._speakers:
        return []
    
    ref = labeler._speakers[label]
    if not ref.embeddings or ref.centroid is None:
        return []
    
    centroid = ref.centroid.flatten()
    centroid_norm = np.linalg.norm(centroid)
    
    embeddings_list = [emb.flatten() for emb in ref.embeddings]
    embeddings_matrix = np.vstack(embeddings_list)
    
    # Cosine similarity = dot / (norm_a * norm_b)
    dots = embeddings_matrix @ centroid
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    sims = dots / (norms * centroid_norm + 1e-10)
    
    # Outlier threshold: 2 std below mean
    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims)) if len(sims) > 1 else 0.0
    outlier_threshold = mean_sim - (2 * std_sim)
    
    # Pre-compute similarity to nearest other speaker's centroid
    other_centroids = {}
    for other_label, other_ref in labeler._speakers.items():
        if other_label != label and other_ref.centroid is not None:
            other_centroids[other_label] = other_ref.centroid.flatten()
    
    nearest_other_sims = None
    if other_centroids:
        other_matrix = np.vstack(list(other_centroids.values()))
        other_norms = np.linalg.norm(other_matrix, axis=1)
        nearest_other_sims = []
        for emb in embeddings_list:
            emb_norm = np.linalg.norm(emb)
            other_dots = emb @ other_matrix.T
            other_sims = other_dots / (emb_norm * other_norms + 1e-10)
            nearest_other_sims.append(float(np.max(other_sims)))
    
    details = []
    for i, sim in enumerate(sims):
        detail = {
            "index": i,
            "cosine_sim_to_centroid": round(float(sim), 4),
            "is_outlier": bool(sim < outlier_threshold) if std_sim > 0 else False,
            "cosine_sim_to_nearest_other": (
                round(nearest_other_sims[i], 4) if nearest_other_sims else None
            ),
        }
        details.append(detail)
    
    return details


# ---------------------------------------------------------------------------
# Helper: resolve optional label and validate speaker existence
# ---------------------------------------------------------------------------

def _resolve_label(labeler, label: Optional[str] = None) -> Optional[str]:
    """Validate and resolve an optional speaker label.
    
    Returns the label if valid, raises 404 if speaker not found,
    or returns None if no label was provided (collection mode).
    """
    if label is None:
        return None
    
    if label not in labeler._speakers:
        available = list(labeler._speakers.keys())
        console.print(f"[warning]Speaker '{label}' not found. Available: {available}[/warning]")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "speaker_not_found",
                "label": label,
                "available_labels": available,
            }
        )
    return label


# ---------------------------------------------------------------------------
# GET /segment/centroid-health-report[/{label}]
# ---------------------------------------------------------------------------

@router.get("/segment/centroid-health-report")
@router.get("/segment/centroid-health-report/{label}")
async def get_segment_centroid_health_report(
    label: Optional[str] = None,
):
    """Get centroid health report — all speakers or a single speaker.
    
    Collection mode (no label):
        Returns full CentroidHealthReport for all speakers with thresholds.
    
    Item mode (with label):
        Returns detailed health metrics for a single speaker including 
        per-embedding details.
    
    Parameters
    ----------
    label : str, optional
        Speaker label (e.g. 'SPEAKER_01'). Omit for all speakers.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    if not hasattr(labeler, 'get_centroid_health_report'):
        raise HTTPException(
            status_code=400,
            detail="Health mixin not available on speaker labeler"
        )
    
    label = _resolve_label(labeler, label)
    
    if label is not None:
        # --- Item Mode: Single Speaker ---
        console.print(f"[info]Fetching segment health report for '{label}'[/info]")
        
        report = labeler.get_centroid_health_report()
        if report is None:
            return JSONResponse(content={
                "label": label,
                "error": "no_speakers",
                "timestamp": datetime.now().isoformat(),
            })
        
        if label not in report.results:
            return JSONResponse(content={
                "label": label,
                "error": "speaker_not_in_report",
                "available_labels": list(report.results.keys()),
                "timestamp": datetime.now().isoformat(),
            })
        
        health = report.results[label]
        ref = labeler._speakers.get(label)
        
        response_data = {
            "label": label,
            "embedding_count": health.embedding_count,
            "flags": [f.name for f in health.flags],
            "is_healthy": health.is_healthy,
            "cohesion": round(health.mean_cosine_sim_to_centroid, 4),
            "spread": round(health.intra_cluster_spread, 4),
            "silhouette": round(health.silhouette_score, 4),
            "nearest_label": health.nearest_centroid_label,
            "nearest_similarity": (
                round(health.nearest_centroid_similarity, 4)
                if health.nearest_centroid_similarity is not None else None
            ),
            "segment_count": ref.segment_count if ref else health.embedding_count,
            "centroid_quality": ref.centroid_quality if ref else 0.0,
            "first_seen": (ref.first_seen or 0.0) if ref else 0.0,
            "last_seen": ref.last_seen if ref else 0.0,
            "embeddings_detail": _compute_per_embedding_details(labeler, label),
            "thresholds": {
                "min_embedding_count": report.thresholds.min_embedding_count,
                "min_mean_cosine_sim": report.thresholds.min_mean_cosine_sim,
                "max_intra_spread": report.thresholds.max_intra_spread,
                "min_silhouette_score": report.thresholds.min_silhouette_score,
                "max_inter_centroid_similarity": report.thresholds.max_inter_centroid_similarity,
                "merge_similarity_threshold": report.thresholds.merge_similarity_threshold,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        console.print(f"[success]Segment health report returned for '{label}' "
                      f"(healthy={health.is_healthy})[/success]")
        return JSONResponse(content=response_data)
    
    else:
        # --- Collection Mode: All Speakers ---
        console.print("[info]Fetching segment health report for all speakers[/info]")
        
        report = labeler.get_centroid_health_report()
        if report is None:
            console.print("[warning]No centroid health report available (no speakers)[/warning]")
            return JSONResponse(content={
                "speaker_count": 0,
                "speakers": {},
                "error": "no_speakers",
                "timestamp": datetime.now().isoformat(),
            })
        
        speakers_data = {}
        for sp_label, health in report.results.items():
            ref = labeler._speakers.get(sp_label)
            speakers_data[sp_label] = {
                "label": sp_label,
                "embedding_count": health.embedding_count,
                "flags": [f.name for f in health.flags],
                "is_healthy": health.is_healthy,
                "cohesion": round(health.mean_cosine_sim_to_centroid, 4),
                "spread": round(health.intra_cluster_spread, 4),
                "silhouette": round(health.silhouette_score, 4),
                "nearest_label": health.nearest_centroid_label,
                "nearest_similarity": (
                    round(health.nearest_centroid_similarity, 4)
                    if health.nearest_centroid_similarity is not None else None
                ),
                "segment_count": ref.segment_count if ref else health.embedding_count,
                "centroid_quality": ref.centroid_quality if ref else 0.0,
            }
        
        response_data = {
            "speaker_count": len(report.results),
            "healthy_count": len(report.healthy_labels),
            "unhealthy_count": len(report.unhealthy_labels),
            "merge_candidate_count": len(report.merge_candidates),
            "merge_candidates": [
                {"label_a": a, "label_b": b, "similarity": round(sim, 4)}
                for a, b, sim in report.merge_candidates
            ],
            "speakers": speakers_data,
            "thresholds": {
                "min_embedding_count": report.thresholds.min_embedding_count,
                "min_mean_cosine_sim": report.thresholds.min_mean_cosine_sim,
                "max_intra_spread": report.thresholds.max_intra_spread,
                "min_silhouette_score": report.thresholds.min_silhouette_score,
                "max_inter_centroid_similarity": report.thresholds.max_inter_centroid_similarity,
                "merge_similarity_threshold": report.thresholds.merge_similarity_threshold,
            },
            "summary": report.summary() if hasattr(report, 'summary') else "",
            "timestamp": datetime.now().isoformat(),
        }
        
        console.print(f"[success]Segment health report returned for "
                      f"{len(report.results)} speakers[/success]")
        return JSONResponse(content=response_data)


# ---------------------------------------------------------------------------
# GET /segment/centroid-health[/{label}]
# ---------------------------------------------------------------------------

@router.get("/segment/centroid-health")
@router.get("/segment/centroid-health/{label}")
async def get_segment_centroid_health(
    label: Optional[str] = None,
):
    """Get centroid health as JSON-serializable dict — all or single speaker.
    
    Collection mode (no label):
        Returns per-speaker health metrics, merge candidates, and thresholds.
    
    Item mode (with label):
        Returns detailed health for one speaker with per-embedding details.
    
    Parameters
    ----------
    label : str, optional
        Speaker label (e.g. 'SPEAKER_01'). Omit for all speakers.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    if not hasattr(labeler, 'get_centroid_health_dict'):
        raise HTTPException(
            status_code=400,
            detail="Health mixin not available on speaker labeler"
        )
    
    label = _resolve_label(labeler, label)
    
    if label is not None:
        # --- Item Mode ---
        console.print(f"[info]Fetching segment health dict for '{label}'[/info]")
        
        full_health = labeler.get_centroid_health_dict()
        speakers = full_health.get("speakers", {})
        
        if label not in speakers:
            return JSONResponse(content={
                "label": label,
                "error": "speaker_not_found",
                "available_labels": list(speakers.keys()),
                "timestamp": datetime.now().isoformat(),
            })
        
        speaker_data = speakers[label]
        speaker_data["embeddings_detail"] = _compute_per_embedding_details(labeler, label)
        speaker_data["timestamp"] = datetime.now().isoformat()
        
        console.print(f"[success]Segment health dict returned for '{label}' "
                      f"({len(speaker_data.get('embeddings_detail', []))} embeddings)[/success]")
        return JSONResponse(content=speaker_data)
    
    else:
        # --- Collection Mode ---
        console.print("[info]Fetching segment health dict for all speakers[/info]")
        
        health_dict = labeler.get_centroid_health_dict()
        health_dict["timestamp"] = datetime.now().isoformat()
        speaker_count = health_dict.get("speaker_count", 0)
        
        console.print(f"[success]Segment health dict returned: {speaker_count} speakers, "
                      f"{health_dict.get('healthy_count', 0)} healthy[/success]")
        return JSONResponse(content=health_dict)


# ---------------------------------------------------------------------------
# GET /segment/similarity-matrix[/{label}]
# ---------------------------------------------------------------------------

@router.get("/segment/similarity-matrix")
@router.get("/segment/similarity-matrix/{label}")
async def get_segment_similarity_matrix(
    label: Optional[str] = None,
):
    """Get pairwise similarity matrix — full matrix or single row.
    
    Collection mode (no label):
        Returns N×N similarity matrix with labels, segment counts, and health flags.
    
    Item mode (with label):
        Returns 1-row matrix: this speaker vs all others with nearest/most-distant
        neighbors. Suitable for horizontal bar charts.
    
    Parameters
    ----------
    label : str, optional
        Speaker label (e.g. 'SPEAKER_01'). Omit for full matrix.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    if not hasattr(labeler, 'get_similarity_matrix_dict'):
        raise HTTPException(
            status_code=400,
            detail="Health mixin not available on speaker labeler"
        )
    
    label = _resolve_label(labeler, label)
    
    if label is not None:
        # --- Item Mode: Single Row ---
        console.print(f"[info]Fetching segment similarity matrix for '{label}'[/info]")
        
        matrix_data = labeler.get_similarity_matrix_dict()
        labels = matrix_data.get("labels", [])
        
        if not labels:
            return JSONResponse(content={
                "label": label,
                "error": "no_speakers",
                "timestamp": datetime.now().isoformat(),
            })
        
        if label not in labels:
            return JSONResponse(content={
                "label": label,
                "error": "speaker_not_in_matrix",
                "available_labels": labels,
                "timestamp": datetime.now().isoformat(),
            })
        
        label_idx = labels.index(label)
        matrix = matrix_data.get("matrix", [])
        segment_counts = matrix_data.get("segment_counts", [])
        flags_per_label = matrix_data.get("flags_per_label", {})
        
        similarities = matrix[label_idx] if label_idx < len(matrix) else []
        
        # Build other-speaker lists (exclude self)
        other_labels = []
        other_similarities = []
        other_segment_counts = []
        other_flags = []
        
        for i, lbl in enumerate(labels):
            if lbl != label:
                other_labels.append(lbl)
                other_similarities.append(round(similarities[i], 4))
                other_segment_counts.append(
                    segment_counts[i] if i < len(segment_counts) else 0
                )
                other_flags.append(flags_per_label.get(lbl, []))
        
        # Find nearest and most distant neighbors
        nearest_idx = None
        most_distant_idx = None
        if other_similarities:
            nearest_idx = max(range(len(other_similarities)), key=lambda i: other_similarities[i])
            most_distant_idx = min(range(len(other_similarities)), key=lambda i: other_similarities[i])
        
        response_data = {
            "label": label,
            "self_similarity": round(similarities[label_idx], 4) if label_idx < len(similarities) else 1.0,
            "other_labels": other_labels,
            "similarities": other_similarities,
            "other_segment_counts": other_segment_counts,
            "other_flags": other_flags,
            "nearest_neighbor": other_labels[nearest_idx] if nearest_idx is not None else None,
            "nearest_similarity": other_similarities[nearest_idx] if nearest_idx is not None else None,
            "most_distant_neighbor": other_labels[most_distant_idx] if most_distant_idx is not None else None,
            "most_distant_similarity": other_similarities[most_distant_idx] if most_distant_idx is not None else None,
            "own_segment_count": segment_counts[label_idx] if label_idx < len(segment_counts) else 0,
            "own_flags": flags_per_label.get(label, []),
            "timestamp": datetime.now().isoformat(),
        }
        
        console.print(f"[success]Segment similarity matrix returned for '{label}' "
                      f"vs {len(other_labels)} others "
                      f"(nearest: {response_data['nearest_neighbor']})[/success]")
        return JSONResponse(content=response_data)
    
    else:
        # --- Collection Mode: Full Matrix ---
        console.print("[info]Fetching segment similarity matrix for all speakers[/info]")
        
        matrix_data = labeler.get_similarity_matrix_dict()
        matrix_data["timestamp"] = datetime.now().isoformat()
        matrix_data["dimension"] = len(matrix_data.get("labels", []))
        matrix_data["has_data"] = len(matrix_data.get("labels", [])) >= 2
        
        label_count = len(matrix_data.get("labels", []))
        console.print(f"[success]Segment similarity matrix returned: "
                      f"{label_count}x{label_count}[/success]")
        return JSONResponse(content=matrix_data)


# ---------------------------------------------------------------------------
# GET /segment/insights[/{label}]
# ---------------------------------------------------------------------------

@router.get("/segment/insights")
@router.get("/segment/insights/{label}")
async def get_segment_insights(
    label: Optional[str] = None,
):
    """Get high-level insights — system overview or single speaker detail.
    
    Collection mode (no label):
        Returns system health, alerts, badges for all speakers, flag summary,
        and top merge candidates. Suitable for dashboard summary cards.
    
    Item mode (with label):
        Returns detailed insights for one speaker including badge, relevant 
        alerts, relationship data, and recommended action. Suitable for 
        a single-speaker detail card.
    
    Parameters
    ----------
    label : str, optional
        Speaker label (e.g. 'SPEAKER_01'). Omit for system overview.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    if not hasattr(labeler, 'get_speaker_insights'):
        raise HTTPException(
            status_code=400,
            detail="Health mixin not available on speaker labeler"
        )
    
    label = _resolve_label(labeler, label)
    
    if label is not None:
        # --- Item Mode ---
        console.print(f"[info]Fetching segment insights for '{label}'[/info]")
        
        insights = labeler.get_speaker_insights()
        health_dict = labeler.get_centroid_health_dict()
        
        speakers = health_dict.get("speakers", {})
        
        # Badge from insights
        badges = insights.get("badges", {})
        badge = badges.get(label, {"label": label, "badge": "❓ Unknown", "color": "grey"})
        
        # Alerts relevant to this speaker
        relevant_alerts = []
        for alert in insights.get("alerts", []):
            if label in alert.get("labels", []):
                relevant_alerts.append({
                    "level": alert.get("level", "info"),
                    "message": alert.get("message", ""),
                })
        
        # Merge candidates involving this speaker
        merge_candidates_with_self = []
        for candidate in health_dict.get("merge_candidates", []):
            if candidate["label_a"] == label:
                merge_candidates_with_self.append({
                    "other_label": candidate["label_b"],
                    "similarity": candidate["similarity"],
                })
            elif candidate["label_b"] == label:
                merge_candidates_with_self.append({
                    "other_label": candidate["label_a"],
                    "similarity": candidate["similarity"],
                })
        
        # Recommended action based on flags
        speaker_data = speakers.get(label, {})
        flags = speaker_data.get("flags", [])
        
        if "CONTAMINATED" in flags:
            recommended_action = "Reset or re-evaluate this speaker's centroid"
        elif "REDUNDANT" in flags:
            nearest = speaker_data.get("nearest_label", "nearest speaker")
            recommended_action = f"Consider merging with {nearest}"
        elif "TOO_CLOSE" in flags:
            recommended_action = "Monitor similarity to nearest neighbor"
        elif "DIFFUSE" in flags:
            recommended_action = "Speaker embeddings are spread out — may need more data"
        elif "IMMATURE" in flags:
            recommended_action = "Collect more segments for reliable identification"
        else:
            recommended_action = "Speaker is healthy — no action needed"
        
        response_data = {
            "label": label,
            "badge": badge,
            "is_healthy": speaker_data.get("is_healthy", False),
            "flags": flags,
            "alerts": relevant_alerts,
            "stats": {
                "embedding_count": speaker_data.get("embedding_count", 0),
                "cohesion": speaker_data.get("cohesion", 0),
                "spread": speaker_data.get("spread", 0),
                "silhouette": speaker_data.get("silhouette", 0),
                "segment_count": speaker_data.get("segment_count", 0),
                "centroid_quality": speaker_data.get("centroid_quality", 0),
            },
            "relationships": {
                "nearest_neighbor": speaker_data.get("nearest_label"),
                "nearest_similarity": speaker_data.get("nearest_similarity"),
                "merge_candidates_with_self": merge_candidates_with_self,
            },
            "recommended_action": recommended_action,
            "timestamp": datetime.now().isoformat(),
        }
        
        console.print(f"[success]Segment insights returned for '{label}' "
                      f"(healthy={speaker_data.get('is_healthy')})[/success]")
        return JSONResponse(content=response_data)
    
    else:
        # --- Collection Mode ---
        console.print("[info]Fetching segment insights for all speakers[/info]")
        
        insights = labeler.get_speaker_insights()
        insights["timestamp"] = datetime.now().isoformat()
        
        if insights.get("system_health") == "critical":
            insights["recommended_action"] = "Review and merge redundant speakers immediately"
        elif insights.get("system_health") == "warning":
            insights["recommended_action"] = "Monitor closely and consider consolidation"
        else:
            insights["recommended_action"] = "System is healthy, no action needed"
        
        total = insights.get("total_speakers", 0)
        healthy = insights.get("healthy_speakers", 0)
        
        console.print(f"[success]Segment insights returned: {healthy}/{total} healthy, "
                      f"system_health={insights.get('system_health')}[/success]")
        return JSONResponse(content=insights)


# ---------------------------------------------------------------------------
# GET /segment/cohesion-series[/{label}]
# ---------------------------------------------------------------------------

@router.get("/segment/cohesion-series")
@router.get("/segment/cohesion-series/{label}")
async def get_segment_cohesion_series(
    label: Optional[str] = None,
):
    """Get cohesion over time — all speakers or single speaker.
    
    Collection mode (no label):
        Returns per-speaker cohesion series with trend summaries and 
        stability overview (degrading/improving/stable counts).
    
    Item mode (with label):
        Returns detailed cohesion series for one speaker with per-embedding
        cosine similarity, outlier detection, and cumulative mean.
        Suitable for a line chart showing centroid stability evolution.
    
    Parameters
    ----------
    label : str, optional
        Speaker label (e.g. 'SPEAKER_01'). Omit for all speakers.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    if not hasattr(labeler, 'get_cohesion_series'):
        raise HTTPException(
            status_code=400,
            detail="Health mixin not available on speaker labeler"
        )
    
    label = _resolve_label(labeler, label)
    
    if label is not None:
        # --- Item Mode ---
        console.print(f"[info]Fetching cohesion series for '{label}'[/info]")
        
        cohesion_data = labeler.get_cohesion_series()
        speakers = cohesion_data.get("speakers", {})
        
        if label not in speakers:
            return JSONResponse(content={
                "label": label,
                "error": "speaker_not_found",
                "available_labels": list(speakers.keys()),
                "timestamp": datetime.now().isoformat(),
            })
        
        speaker_series = speakers[label]
        series_values = speaker_series.get("series", [])
        mean_coh = speaker_series.get("mean_cohesion", 0)
        
        # Outlier detection: 2 std below mean
        std_coh = float(np.std(series_values)) if len(series_values) > 1 else 0.0
        outlier_threshold = mean_coh - (2 * std_coh) if std_coh > 0 else -1.0
        
        # Build detailed series with cumulative mean
        detailed_series = []
        cumulative_sum = 0.0
        for i, sim in enumerate(series_values):
            cumulative_sum += sim
            cumulative_mean = cumulative_sum / (i + 1)
            detailed_series.append({
                "index": i,
                "cosine_sim_to_centroid": round(sim, 4),
                "is_outlier": bool(sim < outlier_threshold) if std_coh > 0 else False,
                "cumulative_mean": round(cumulative_mean, 4),
            })
        
        response_data = {
            "label": label,
            "series": detailed_series,
            "stats": {
                "mean_cohesion": round(mean_coh, 4),
                "min_cohesion": round(min(series_values), 4) if series_values else 0,
                "max_cohesion": round(max(series_values), 4) if series_values else 0,
                "std_cohesion": round(std_coh, 4),
                "trend": speaker_series.get("trend", "stable"),
                "outlier_count": sum(1 for d in detailed_series if d["is_outlier"]),
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        console.print(f"[success]Cohesion series returned for '{label}' "
                      f"({len(detailed_series)} points, "
                      f"trend={speaker_series.get('trend')})[/success]")
        return JSONResponse(content=response_data)
    
    else:
        # --- Collection Mode ---
        console.print("[info]Fetching cohesion series for all speakers[/info]")
        
        cohesion_data = labeler.get_cohesion_series()
        speakers_data = cohesion_data.get("speakers", {})
        
        trends = {}
        for sp_label, data in speakers_data.items():
            trends[sp_label] = data.get("trend", "stable")
        
        degrading_count = sum(1 for t in trends.values() if t == "degrading")
        improving_count = sum(1 for t in trends.values() if t == "improving")
        stable_count = sum(1 for t in trends.values() if t == "stable")
        
        cohesion_data["timestamp"] = datetime.now().isoformat()
        cohesion_data["speaker_count"] = len(speakers_data)
        cohesion_data["trends_summary"] = trends
        cohesion_data["stability_overview"] = {
            "degrading": degrading_count,
            "improving": improving_count,
            "stable": stable_count,
            "overall_assessment": (
                "stable" if degrading_count == 0 
                else ("warning" if degrading_count <= 2 else "concerning")
            ),
        }
        
        console.print(f"[success]Cohesion series returned: {len(speakers_data)} speakers, "
                      f"degrading={degrading_count}, improving={improving_count}, "
                      f"stable={stable_count}[/success]")
        return JSONResponse(content=cohesion_data)


# ---------------------------------------------------------------------------
# GET /segment/chart-data[/{label}]
# ---------------------------------------------------------------------------

@router.get("/segment/chart-data")
@router.get("/segment/chart-data/{label}")
async def get_segment_chart_data(
    label: Optional[str] = None,
):
    """Get all chart-ready payloads — aggregated for system or single speaker.
    
    Collection mode (no label):
        Returns health, similarity matrix, insights, cohesion series, and 
        summary text for all speakers. Recommended for initial dashboard loads.
    
    Item mode (with label):
        Returns all chart payloads for a single speaker in one call.
        Recommended for single-speaker detail pages to minimize round trips.
    
    Parameters
    ----------
    label : str, optional
        Speaker label (e.g. 'SPEAKER_01'). Omit for all speakers.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        raise HTTPException(status_code=400, detail="Speaker labeler not initialized")
    
    label = _resolve_label(labeler, label)
    
    if label is not None:
        # --- Item Mode ---
        console.print(f"[info]Fetching aggregated chart data for '{label}'[/info]")
        
        chart_data = {
            "label": label,
            "health": None,
            "similarity": None,
            "insights": None,
            "cohesion": None,
            "summary_text": "",
            "timestamp": datetime.now().isoformat(),
        }
        
        # Health
        if hasattr(labeler, 'get_centroid_health_dict'):
            full_health = labeler.get_centroid_health_dict()
            speakers = full_health.get("speakers", {})
            if label in speakers:
                chart_data["health"] = speakers[label]
                chart_data["health"]["embeddings_detail"] = _compute_per_embedding_details(
                    labeler, label
                )
        
        # Similarity (single row)
        if hasattr(labeler, 'get_similarity_matrix_dict'):
            matrix_data = labeler.get_similarity_matrix_dict()
            labels = matrix_data.get("labels", [])
            if label in labels:
                label_idx = labels.index(label)
                matrix = matrix_data.get("matrix", [])
                segment_counts = matrix_data.get("segment_counts", [])
                flags_per_label = matrix_data.get("flags_per_label", {})
                
                similarities = matrix[label_idx] if label_idx < len(matrix) else []
                
                other_labels = []
                other_sims = []
                for i, lbl in enumerate(labels):
                    if lbl != label:
                        other_labels.append(lbl)
                        other_sims.append(round(similarities[i], 4))
                
                nearest_idx = max(range(len(other_sims)), key=lambda i: other_sims[i]) if other_sims else None
                most_distant_idx = min(range(len(other_sims)), key=lambda i: other_sims[i]) if other_sims else None
                
                chart_data["similarity"] = {
                    "other_labels": other_labels,
                    "similarities": other_sims,
                    "nearest_neighbor": other_labels[nearest_idx] if nearest_idx is not None else None,
                    "nearest_similarity": other_sims[nearest_idx] if nearest_idx is not None else None,
                    "most_distant_neighbor": other_labels[most_distant_idx] if most_distant_idx is not None else None,
                    "most_distant_similarity": other_sims[most_distant_idx] if most_distant_idx is not None else None,
                    "own_segment_count": segment_counts[label_idx] if label_idx < len(segment_counts) else 0,
                    "own_flags": flags_per_label.get(label, []),
                }
        
        # Insights
        if hasattr(labeler, 'get_speaker_insights'):
            insights = labeler.get_speaker_insights()
            badges = insights.get("badges", {})
            badge = badges.get(label, {"label": label, "badge": "❓ Unknown", "color": "grey"})
            
            relevant_alerts = []
            for alert in insights.get("alerts", []):
                if label in alert.get("labels", []):
                    relevant_alerts.append({
                        "level": alert.get("level", "info"),
                        "message": alert.get("message", ""),
                    })
            
            chart_data["insights"] = {
                "badge": badge,
                "alerts": relevant_alerts,
                "is_healthy": (
                    chart_data["health"].get("is_healthy", False) 
                    if chart_data["health"] else False
                ),
                "flags": (
                    chart_data["health"].get("flags", []) 
                    if chart_data["health"] else []
                ),
            }
        
        # Cohesion
        if hasattr(labeler, 'get_cohesion_series'):
            cohesion_data = labeler.get_cohesion_series()
            speakers = cohesion_data.get("speakers", {})
            if label in speakers:
                speaker_cohesion = speakers[label]
                series_values = speaker_cohesion.get("series", [])
                chart_data["cohesion"] = {
                    "series": series_values,
                    "mean_cohesion": speaker_cohesion.get("mean_cohesion", 0),
                    "min_cohesion": min(series_values) if series_values else 0,
                    "max_cohesion": max(series_values) if series_values else 0,
                    "trend": speaker_cohesion.get("trend", "stable"),
                }
        
        # Summary text
        health = chart_data.get("health", {})
        is_healthy = health.get("is_healthy", False) if health else False
        flags = health.get("flags", []) if health else []
        cohesion_val = health.get("cohesion", 0) if health else 0
        
        if is_healthy:
            chart_data["summary_text"] = (
                f"Speaker {label} is HEALTHY (cohesion: {cohesion_val:.3f})"
            )
        elif flags:
            chart_data["summary_text"] = (
                f"Speaker {label} needs attention: {', '.join(flags[:3])}"
            )
        else:
            chart_data["summary_text"] = f"Speaker {label} status unknown"
        
        console.print(f"[success]Chart data aggregated for '{label}'[/success]")
        return JSONResponse(content=chart_data)
    
    else:
        # --- Collection Mode ---
        console.print("[info]Fetching aggregated chart data for all speakers[/info]")
        
        if not hasattr(labeler, 'get_chart_data'):
            # Aggregate individually
            chart_data = {
                "health": None,
                "similarity": None,
                "insights": None,
                "cohesion": None,
                "summary_text": "",
                "timestamp": datetime.now().isoformat(),
                "aggregated": True,
            }
            
            if hasattr(labeler, 'get_centroid_health_dict'):
                chart_data["health"] = labeler.get_centroid_health_dict()
            if hasattr(labeler, 'get_similarity_matrix_dict'):
                chart_data["similarity"] = labeler.get_similarity_matrix_dict()
            if hasattr(labeler, 'get_speaker_insights'):
                chart_data["insights"] = labeler.get_speaker_insights()
            if hasattr(labeler, 'get_cohesion_series'):
                chart_data["cohesion"] = labeler.get_cohesion_series()
            
            if chart_data["insights"]:
                ins = chart_data["insights"]
                total = ins.get("total_speakers", 0)
                healthy = ins.get("healthy_speakers", 0)
                system = ins.get("system_health", "unknown")
                chart_data["summary_text"] = (
                    f"System Health: {system.upper()} — {healthy}/{total} speakers healthy"
                )
            
            console.print(f"[success]Chart data aggregated individually: "
                          f"{chart_data.get('health', {}).get('speaker_count', 0)} speakers[/success]")
            return JSONResponse(content=chart_data)
        
        chart_data = labeler.get_chart_data()
        chart_data["timestamp"] = datetime.now().isoformat()
        health = chart_data.get("health", {})
        speaker_count = health.get("speaker_count", 0) if health else 0
        
        console.print(f"[success]Chart data returned: {speaker_count} speakers, "
                      f"{'summary' in chart_data} sections[/success]")
        return JSONResponse(content=chart_data)


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
