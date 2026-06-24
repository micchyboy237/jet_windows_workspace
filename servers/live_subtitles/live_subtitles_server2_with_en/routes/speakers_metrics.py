"""
Speaker metrics and health dashboard routes.

Provides API endpoints and HTML pages for:
    - Intra-speaker cohesion metrics
    - Inter-speaker separation metrics
    - Segment group health (labeling quality)
    - Outlier pool health
    - Overall speaker system health summary
"""

from datetime import datetime
from typing import Optional

from core.state import get_speaker_labeler
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
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
    autoescape=select_autoescape(["html", "xml"]),
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
            status_code=500, detail=f"Template {name} not found or invalid"
        )


def render_template(name: str, context: dict = None) -> str:
    """Render a template with context."""
    template = get_template(name)
    return template.render(**(context or {}))


def _check_labeler():
    """Verify labeler is initialized and has metrics mixin."""
    labeler = get_speaker_labeler()
    if not labeler:
        console.print("[error]Speaker labeler not initialized[/]")
        raise HTTPException(
            status_code=400,
            detail="Speaker labeler not initialized. Process some audio segments first.",
        )
    if not hasattr(labeler, "get_speaker_metrics"):
        console.print("[error]SpeakerMetricsMixin not applied to labeler[/]")
        raise HTTPException(
            status_code=500,
            detail="Speaker metrics not available. Mixin not applied.",
        )
    return labeler


# ============================================================
# COMBINED METRICS (primary data endpoint)
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

    Returns intra-speaker cohesion, inter-speaker separation, segment group
    health, and outlier pool health in a single response.

    Parameters
    ----------
    label : str, optional
        Specific speaker label to filter intra-speaker results.

    Returns
    -------
    JSON with structure:
    {
        "intra_speaker": { speakers, overall_status, total_speakers_analyzed },
        "inter_speaker": { meanSeparation, health, pairwise, closest_pair, ... },
        "segment_groups": { totalSegments, outlierRatio, labelStability, timeline, ... },
        "outliers": { activeCount, totalPromotions, outlierDetails, ... },
        "summary": { overall, intra, inter, segmentGroups, outliers, totalSpeakers, ... },
        "timestamp": "ISO datetime"
    }
    """
    labeler = _check_labeler()

    console.print(
        f"[info]📊 Fetching full speaker metrics data (label={label or 'all'})[/]"
    )
    try:
        metrics_data = labeler.get_speaker_metrics(label=label)
        
        # Log summary for quick diagnosis
        summary = metrics_data.get("summary", {})
        console.print(
            f"[success]✅ Speaker metrics: "
            f"overall={summary.get('overall', '?')}, "
            f"speakers={summary.get('totalSpeakers', 0)}, "
            f"segments={summary.get('totalSegments', 0)}, "
            f"outliers={summary.get('outlierCount', 0)}[/]"
        )
        return JSONResponse(content=metrics_data)
    except Exception as e:
        console.print(f"[error]Failed to compute speaker metrics: {e}[/]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/]")
        raise HTTPException(
            status_code=500, detail=f"Failed to compute speaker metrics: {str(e)}"
        )


# ============================================================
# INDIVIDUAL METRICS ENDPOINTS
# ============================================================


@router.get("/metrics/intra")
async def get_intra_speaker_metrics(
    label: Optional[str] = Query(None, description="Filter to specific speaker label"),
):
    """
    Get intra-speaker cohesion metrics only.
    
    Returns per-speaker similarity to centroid, silhouette scores,
    and segment-level detail for timeline visualization.
    """
    labeler = _check_labeler()

    console.print(f"[info]📊 Fetching intra-speaker metrics (label={label or 'all'})[/]")
    result = labeler.compute_intra_speaker_metrics(label=label)
    return JSONResponse(content=result)


@router.get("/metrics/inter")
async def get_inter_speaker_metrics():
    """
    Get inter-speaker separation metrics only.
    
    Returns pairwise distances between speaker centroids, closest pair
    identification, and distance matrix data.
    """
    labeler = _check_labeler()

    console.print("[info]📊 Fetching inter-speaker metrics[/]")
    result = labeler.compute_inter_speaker_metrics()
    return JSONResponse(content=result)


@router.get("/metrics/segments")
async def get_segment_group_health():
    """
    Get segment group health metrics.
    
    Shows labeling quality: outlier ratio, rejection rate, label stability,
    and match type distribution over time.
    """
    labeler = _check_labeler()

    console.print("[info]📊 Fetching segment group health metrics[/]")
    result = labeler.compute_segment_group_health()
    return JSONResponse(content=result)


@router.get("/metrics/outliers")
async def get_outlier_health():
    """
    Get outlier pool health metrics.
    
    Shows active outliers, promotion history, and pool statistics.
    """
    labeler = _check_labeler()

    console.print("[info]📊 Fetching outlier health metrics[/]")
    result = labeler.compute_outlier_health()
    return JSONResponse(content=result)


@router.get("/metrics/segment-detail/{segment_id}")
async def get_segment_detail(segment_id: str):
    """
    Get detailed information about a specific segment.
    
    Parameters
    ----------
    segment_id : str
        The unique segment identifier (e.g., 'segment_a3f2b1c4')
    """
    labeler = _check_labeler()

    console.print(f"[info]🔍 Fetching segment detail for '{segment_id}'[/]")
    result = labeler.get_segment_detail(segment_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Segment '{segment_id}' not found",
        )
    return JSONResponse(content=result)


# ============================================================
# HEALTH SUMMARY
# ============================================================


@router.get("/metrics/health")
async def get_speaker_metrics_health():
    """
    Quick health check endpoint for speaker metrics.
    
    Returns a simple summary of overall speaker system health
    suitable for monitoring/alerting.
    """
    labeler = _check_labeler()

    metrics = labeler.get_speaker_metrics()
    summary = metrics.get("summary", {})
    
    return JSONResponse(
        content={
            "overall": summary.get("overall", "unknown"),
            "intra_status": summary.get("intra", "unknown"),
            "inter_health": summary.get("inter", "unknown"),
            "segment_groups_health": summary.get("segmentGroups", "unknown"),
            "outlier_health": summary.get("outliers", "unknown"),
            "total_speakers": summary.get("totalSpeakers", 0),
            "total_segments": summary.get("totalSegments", 0),
            "active_outliers": summary.get("outlierCount", 0),
            "timestamp": metrics.get("timestamp", datetime.now().isoformat()),
        }
    )


# ============================================================
# HTML PAGES
# ============================================================


@router.get("/metrics", response_class=HTMLResponse)
async def get_speaker_metrics_page(request: Request):
    """
    Serve the speaker metrics HTML dashboard page.
    
    This returns the speaker_metrics.html template as a complete page.
    The frontend JavaScript will then fetch data from /speakers/metrics/data
    and render the interactive dashboard.
    """
    labeler = get_speaker_labeler()
    if not labeler:
        console.print("[warning]Speaker labeler not initialized for metrics page[/]")
    
    console.print("[info]🖥️  Serving speaker metrics HTML page[/]")
    try:
        html_content = render_template(
            "speaker_metrics.html",
            {
                "title": "Speaker Metrics Dashboard",
                "timestamp": datetime.now().isoformat(),
                "has_labeler": labeler is not None,
            },
        )
        console.print("[success]Speaker metrics page rendered[/]")
        return HTMLResponse(content=html_content)
    except Exception as e:
        console.print(f"[error]Failed to render speaker metrics page: {e}[/]")
        # Fallback to raw HTML file
        metrics_html_path = _templates_dir / "speaker_metrics.html"
        if metrics_html_path.exists():
            console.print("[info]Falling back to raw HTML file[/]")
            return HTMLResponse(content=metrics_html_path.read_text(encoding="utf-8"))
        raise HTTPException(
            status_code=500, detail=f"Failed to render metrics page: {str(e)}"
        )


@router.get("segments2", response_class=HTMLResponse)
async def get_segments_page(request: Request):
    """
    Serve the segments overview HTML page.
    
    Shows segment group history, label timeline, and outlier tracking.
    """
    console.print("[info]🖥️  Serving segments overview HTML page[/]")
    try:
        html_content = render_template(
            "segments.html",
            {
                "title": "Segments Overview",
                "timestamp": datetime.now().isoformat(),
            },
        )
        return HTMLResponse(content=html_content)
    except Exception as e:
        console.print(f"[error]Failed to render segments page: {e}[/]")
        raise HTTPException(
            status_code=500, detail=f"Failed to render segments page: {str(e)}"
        )
