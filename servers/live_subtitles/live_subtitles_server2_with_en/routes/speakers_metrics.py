# Jet_Windows_Workspace/servers/live_subtitles/live_subtitles_server2_with_en/routes/speakers_metrics.py

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
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from services.config import TEMPLATES_DIR

console = Console()

router = APIRouter(prefix="/speakers/metrics", tags=["metrics"])

_templates_dir = TEMPLATES_DIR / "speakers" / "metrics"
_templates_dir.mkdir(parents=True, exist_ok=True)
_jinja_env = Environment(
    loader=FileSystemLoader(str(_templates_dir)),
    autoescape=select_autoescape(["html", "xml"]),
)
console.print(f"[info]Speaker metrics templates directory: {_templates_dir}[/info]")


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


# ═══════════════════════════════════════════════════════════════════════
# HTML Endpoints (Browser-friendly pages)
# ═══════════════════════════════════════════════════════════════════════

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def metrics_overview_html(request: Request):
    """Main metrics overview page with system health summary."""
    labeler = _check_labeler()
    
    try:
        metrics = labeler.get_speaker_metrics()
        console.print("[info]Rendering metrics overview page[/]")
        
        return render_template("overview.html", {
            "active_page": "overview",
            "metrics": metrics,
            "computed_at": metrics.get("computed_at", datetime.now().isoformat()),
            "request": request,
        })
    except Exception as e:
        console.print(f"[error]Error computing metrics: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics: {e}")


@router.get("/speakers", response_class=HTMLResponse)
async def speakers_list_html(request: Request):
    """List all speakers with their cohesion metrics."""
    labeler = _check_labeler()
    
    try:
        cohesion = labeler.get_all_speakers_cohesion()
        console.print(f"[info]Rendering speakers list page: {cohesion.get('total_speakers', 0)} speakers[/]")
        
        return render_template("speakers_list.html", {
            "active_page": "speakers",
            "total_speakers": cohesion.get("total_speakers", 0),
            "average_cohesion": cohesion.get("average_cohesion_score", 0),
            "healthy_count": cohesion.get("healthy_count", 0),
            "warning_count": cohesion.get("warning_count", 0),
            "critical_count": cohesion.get("critical_count", 0),
            "speakers": cohesion.get("speakers", {}),
            "computed_at": cohesion.get("computed_at", datetime.now().isoformat()),
            "request": request,
        })
    except Exception as e:
        console.print(f"[error]Error computing speaker cohesion: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute speaker cohesion: {e}")


@router.get("/speakers/{speaker_label}", response_class=HTMLResponse)
async def speaker_detail_html(request: Request, speaker_label: str):
    """Detailed view for a single speaker with cohesion metrics and segment list."""
    labeler = _check_labeler()
    
    try:
        speaker = labeler.get_speaker_cohesion(speaker_label)
        if not speaker:
            raise HTTPException(status_code=404, detail=f"Speaker '{speaker_label}' not found")
        
        segments = labeler.get_speaker_segment_list(
            speaker_label=speaker_label,
            limit=100,
            offset=0,
        )
        
        console.print(f"[info]Rendering speaker detail page for {speaker_label} with {segments.get('total', 0)} segments[/]")
        
        return render_template("speaker_detail.html", {
            "active_page": "speakers",
            "speaker": speaker,
            "segments": segments,
            "computed_at": datetime.now().isoformat(),
            "request": request,
        })
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Error computing speaker detail for {speaker_label}: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute speaker detail: {e}")


@router.get("/segments", response_class=HTMLResponse)
async def segments_list_html(request: Request):
    """Segment group health page showing labeling quality metrics."""
    labeler = _check_labeler()
    
    try:
        health = labeler.get_segment_group_health()
        console.print(f"[info]Rendering segments health page: {health.get('total_segments', 0)} segments[/]")
        
        return render_template("segments_list.html", {
            "active_page": "segments",
            "health": health,
            "computed_at": health.get("computed_at", datetime.now().isoformat()),
            "request": request,
        })
    except Exception as e:
        console.print(f"[error]Error computing segment health: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute segment health: {e}")


@router.get("/segments/{segment_index}", response_class=HTMLResponse)
async def segment_detail_html(request: Request, segment_index: int):
    """
    Detailed view for a single segment showing all matches and context.
    The segment_index corresponds to the position in _segment_groups (0-based).
    """
    labeler = _check_labeler()
    
    try:
        detail = labeler.get_segment_detail(segment_index)
        if not detail:
            raise HTTPException(status_code=404, detail=f"Segment at index {segment_index} not found")
        
        console.print(f"[info]Rendering segment detail page for index {segment_index}[/]")
        
        return render_template("segment_detail.html", {
            "active_page": "segments",
            "segment": detail,
            "total_segments": len(labeler._segment_groups),
            "computed_at": datetime.now().isoformat(),
            "request": request,
        })
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Error computing segment detail for {segment_index}: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute segment detail: {e}")


@router.get("/separation", response_class=HTMLResponse)
async def separation_html(request: Request):
    """Inter-speaker separation metrics page."""
    labeler = _check_labeler()
    
    try:
        separation = labeler.get_speaker_separation_matrix()
        console.print(f"[info]Rendering separation page: {separation.get('total_speakers_with_centroids', 0)} speakers[/]")
        
        return render_template("separation.html", {
            "active_page": "separation",
            "separation": separation,
            "computed_at": separation.get("computed_at", datetime.now().isoformat()),
            "request": request,
        })
    except Exception as e:
        console.print(f"[error]Error computing separation metrics: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute separation metrics: {e}")


@router.get("/outliers", response_class=HTMLResponse)
async def outliers_html(request: Request):
    """Outlier pool health page."""
    labeler = _check_labeler()
    
    try:
        outlier_health = labeler.get_outlier_pool_health()
        console.print(f"[info]Rendering outlier pool page: enabled={outlier_health.get('enabled')}[/]")
        
        return render_template("outliers.html", {
            "active_page": "outliers",
            "outlier_health": outlier_health,
            "computed_at": outlier_health.get("computed_at", datetime.now().isoformat()),
            "request": request,
        })
    except Exception as e:
        console.print(f"[error]Error computing outlier health: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute outlier health: {e}")


@router.get("/timeline", response_class=HTMLResponse)
async def timeline_html(request: Request):
    """Speaker activity timeline visualization."""
    labeler = _check_labeler()
    
    try:
        timeline = labeler.get_speaker_timeline()
        console.print(f"[info]Rendering timeline page: {timeline.get('total_segments', 0)} segments[/]")
        
        return render_template("timeline.html", {
            "active_page": "timeline",
            "timeline": timeline,
            "computed_at": timeline.get("computed_at", datetime.now().isoformat()),
            "request": request,
        })
    except Exception as e:
        console.print(f"[error]Error computing timeline: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute timeline: {e}")


# ═══════════════════════════════════════════════════════════════════════
# JSON Endpoints (API responses)
# ═══════════════════════════════════════════════════════════════════════

@router.get("/api/overview")
async def metrics_overview_json():
    """JSON endpoint: Full system health overview."""
    labeler = _check_labeler()
    try:
        metrics = labeler.get_speaker_metrics()
        console.print("[info]Returning metrics overview JSON[/]")
        return JSONResponse(content=metrics)
    except Exception as e:
        console.print(f"[error]Error in metrics_overview_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/speakers")
async def speakers_list_json():
    """JSON endpoint: All speakers cohesion metrics."""
    labeler = _check_labeler()
    try:
        cohesion = labeler.get_all_speakers_cohesion()
        console.print(f"[info]Returning speakers cohesion JSON: {cohesion.get('total_speakers', 0)} speakers[/]")
        return JSONResponse(content=cohesion)
    except Exception as e:
        console.print(f"[error]Error in speakers_list_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/speakers/{speaker_label}")
async def speaker_detail_json(speaker_label: str):
    """JSON endpoint: Single speaker cohesion detail."""
    labeler = _check_labeler()
    try:
        speaker = labeler.get_speaker_cohesion(speaker_label)
        if not speaker:
            raise HTTPException(status_code=404, detail=f"Speaker '{speaker_label}' not found")
        console.print(f"[info]Returning speaker detail JSON for {speaker_label}[/]")
        return JSONResponse(content=speaker)
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Error in speaker_detail_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/speakers/{speaker_label}/segments")
async def speaker_segments_json(
    speaker_label: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """JSON endpoint: Segments for a specific speaker (paginated)."""
    labeler = _check_labeler()
    try:
        segments = labeler.get_speaker_segment_list(
            speaker_label=speaker_label,
            limit=limit,
            offset=offset,
        )
        console.print(f"[info]Returning speaker segments JSON for {speaker_label}: {segments.get('total', 0)} total[/]")
        return JSONResponse(content=segments)
    except Exception as e:
        console.print(f"[error]Error in speaker_segments_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/segments")
async def segments_list_json():
    """JSON endpoint: Segment group health metrics."""
    labeler = _check_labeler()
    try:
        health = labeler.get_segment_group_health()
        console.print(f"[info]Returning segments health JSON: {health.get('total_segments', 0)} segments[/]")
        return JSONResponse(content=health)
    except Exception as e:
        console.print(f"[error]Error in segments_list_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/segments/{segment_index}")
async def segment_detail_json(segment_index: int):
    """JSON endpoint: Single segment detail."""
    labeler = _check_labeler()
    try:
        detail = labeler.get_segment_detail(segment_index)
        if not detail:
            raise HTTPException(status_code=404, detail=f"Segment at index {segment_index} not found")
        console.print(f"[info]Returning segment detail JSON for index {segment_index}[/]")
        return JSONResponse(content=detail)
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[error]Error in segment_detail_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/separation")
async def separation_json():
    """JSON endpoint: Inter-speaker separation matrix."""
    labeler = _check_labeler()
    try:
        separation = labeler.get_speaker_separation_matrix()
        console.print(f"[info]Returning separation JSON: {separation.get('total_speakers_with_centroids', 0)} speakers[/]")
        return JSONResponse(content=separation)
    except Exception as e:
        console.print(f"[error]Error in separation_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/outliers")
async def outliers_json():
    """JSON endpoint: Outlier pool health."""
    labeler = _check_labeler()
    try:
        outlier_health = labeler.get_outlier_pool_health()
        console.print(f"[info]Returning outlier pool JSON[/]")
        return JSONResponse(content=outlier_health)
    except Exception as e:
        console.print(f"[error]Error in outliers_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/timeline")
async def timeline_json():
    """JSON endpoint: Speaker activity timeline."""
    labeler = _check_labeler()
    try:
        timeline = labeler.get_speaker_timeline()
        console.print(f"[info]Returning timeline JSON: {timeline.get('total_segments', 0)} segments[/]")
        return JSONResponse(content=timeline)
    except Exception as e:
        console.print(f"[error]Error in timeline_json: {e}[/]")
        raise HTTPException(status_code=500, detail=str(e))
