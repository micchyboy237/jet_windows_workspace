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
from scipy.spatial.distance import cdist
from rich.console import Console
from services.config import TEMPLATES_DIR
import os
import re
from pathlib import Path
from fastapi.responses import StreamingResponse
from services.segment_utils import get_audio_files
from services.config import SEGMENT_AUDIO_DIR

console = Console()
router = APIRouter(prefix="/speakers/metrics", tags=["metrics"])

# Update template directory to use .jinja files
_templates_dir = TEMPLATES_DIR / "speakers" / "metrics"
_templates_dir.mkdir(parents=True, exist_ok=True)

# Configure Jinja2 to use .jinja extension for autoescape
_jinja_env = Environment(
    loader=FileSystemLoader(str(_templates_dir)),
    autoescape=select_autoescape(['html', 'xml', 'jinja', 'jinja2']),
    extensions=['jinja2.ext.debug', 'jinja2.ext.loopcontrols'],
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

# ... (rest of the helper functions stay the same)

# Audio streaming utility functions
def resolve_audio_path(file_param: str) -> str:
    """
    Resolve and validate the audio file path.
    Prevents directory traversal attacks.
    """
    requested_path = Path(file_param)
    safe_path = Path(*[p for p in requested_path.parts if p != '..'])
    full_path = Path(SEGMENT_AUDIO_DIR) / safe_path
    
    try:
        full_path.resolve().relative_to(Path(SEGMENT_AUDIO_DIR).resolve())
    except ValueError:
        console.print(f"[error]Path traversal attempt: {file_param}[/error]")
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not full_path.exists():
        console.print(f"[error]Audio file not found: {full_path}[/error]")
        raise HTTPException(status_code=404, detail="File not found")
    
    return str(full_path)

def get_range_info(range_header: Optional[str], file_size: int) -> tuple[int, int]:
    """Parse the Range header and return start and end bytes."""
    if not range_header:
        return 0, file_size - 1
    
    range_match = re.search(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        console.print(f"[warning]Invalid Range header format: {range_header}[/warning]")
        return 0, file_size - 1
    
    start = int(range_match.group(1))
    end_str = range_match.group(2)
    end = int(end_str) if end_str else file_size - 1
    
    if start >= file_size or end >= file_size or start > end:
        console.print(f"[warning]Invalid range: {start}-{end} for file size {file_size}[/warning]")
        raise HTTPException(status_code=416, detail="Range Not Satisfiable")
    
    return start, end

def generate_audio_chunks(file_path: str, start: int, end: int, chunk_size: int = 65536):
    """Generator that yields audio file chunks for streaming."""
    console.print(f"[dim]Streaming bytes {start}-{end} with chunk size {chunk_size}[/dim]")
    with open(file_path, "rb") as f:
        f.seek(start)
        bytes_remaining = end - start + 1
        while bytes_remaining > 0:
            read_size = min(chunk_size, bytes_remaining)
            chunk = f.read(read_size)
            if not chunk:
                break
            bytes_remaining -= len(chunk)
            yield chunk
    console.print(f"[dim]Finished streaming - {end - start + 1} bytes sent[/dim]")

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

# Update all route handlers to use .jinja extension
@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def metrics_overview_html(request: Request):
    """Main metrics overview page with system health summary."""
    labeler = _check_labeler()
    try:
        metrics = labeler.get_speaker_metrics()
        console.print("[info]Rendering metrics overview page[/]")
        return render_template("overview.jinja", {
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
        return render_template("speakers_list.jinja", {
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
        return render_template("speaker_detail.jinja", {
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
        return render_template("segments_list.jinja", {
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
        return render_template("segment_detail.jinja", {
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
        return render_template("separation.jinja", {
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
    """
    Outlier pool health page with:
    - Summary stats (active, promotions, stale)
    - ALWAYS show list of active outliers (even when empty)
    - Per-outlier nearest speaker centroid analysis
    - Outlier-to-outlier distance matrix
    - Promotion history
    """
    labeler = _check_labeler()
    try:
        outlier_health = labeler.get_outlier_pool_health()
        active_outlier_details = []
        outlier_distance_matrix = {}
        if hasattr(labeler, 'outlier_pool') and labeler.use_outlier_buffer:
            try:
                active_outliers_dict = labeler.outlier_pool._outliers
                if active_outliers_dict:
                    speaker_centroids = {}
                    for sp_label, ref in labeler._speakers.items():
                        if ref.has_valid_centroid:
                            speaker_centroids[sp_label] = {
                                'centroid': ref.centroid,
                                'segment_count': ref.segment_count,
                                'centroid_quality': ref.centroid_quality,
                            }
                    current_time = datetime.now().timestamp()
                    for label, entry in active_outliers_dict.items():
                        embedding = entry.embedding
                        if embedding is None:
                            continue
                        if embedding.ndim == 1:
                            embedding = embedding.reshape(1, -1)
                        age = current_time - entry.timestamp
                        nearest_speakers = []
                        for sp_label, sp_data in speaker_centroids.items():
                            centroid_2d = sp_data['centroid'].reshape(1, -1) if sp_data['centroid'].ndim == 1 else sp_data['centroid']
                            dist = float(cdist(embedding, centroid_2d, metric='cosine')[0, 0])
                            nearest_speakers.append({
                                'speaker_label': sp_label,
                                'cosine_distance': round(dist, 4),
                                'cosine_similarity': round(1.0 - dist, 4),
                                'speaker_segments': sp_data['segment_count'],
                                'centroid_quality': sp_data['centroid_quality'],
                            })
                        nearest_speakers.sort(key=lambda x: x['cosine_distance'])
                        nearest_outliers = []
                        for other_label, other_entry in active_outliers_dict.items():
                            if other_label == label:
                                continue
                            other_emb = other_entry.embedding
                            if other_emb is None:
                                continue
                            if other_emb.ndim == 1:
                                other_emb = other_emb.reshape(1, -1)
                            dist = float(cdist(embedding, other_emb, metric='cosine')[0, 0])
                            nearest_outliers.append({
                                'label': other_label,
                                'cosine_distance': round(dist, 4),
                            })
                        nearest_outliers.sort(key=lambda x: x['cosine_distance'])
                        if nearest_speakers and nearest_speakers[0]['cosine_distance'] < 0.2:
                            promotion_likelihood = 'high'
                        elif nearest_outliers and nearest_outliers[0]['cosine_distance'] < 0.15:
                            promotion_likelihood = 'high'
                        elif nearest_outliers and nearest_outliers[0]['cosine_distance'] < 0.3:
                            promotion_likelihood = 'medium'
                        else:
                            promotion_likelihood = 'low'
                        active_outlier_details.append({
                            'label': label,
                            'age_seconds': round(age, 1),
                            'segment_id': entry.segment_id,
                            'timestamp': entry.timestamp,
                            'audio_duration': entry.audio_duration,
                            'match_attempts': entry.match_attempts,
                            'nearest_speakers': nearest_speakers[:5],
                            'nearest_outliers': nearest_outliers[:5],
                            'promotion_likelihood': promotion_likelihood,
                        })
                    detail_labels = [d['label'] for d in active_outlier_details]
                    for i, label_a in enumerate(detail_labels):
                        outlier_distance_matrix[label_a] = {}
                        entry_a = active_outliers_dict[label_a]
                        emb_a = entry_a.embedding
                        if emb_a.ndim == 1:
                            emb_a = emb_a.reshape(1, -1)
                        for j, label_b in enumerate(detail_labels):
                            if i == j:
                                continue
                            entry_b = active_outliers_dict[label_b]
                            emb_b = entry_b.embedding
                            if emb_b.ndim == 1:
                                emb_b = emb_b.reshape(1, -1)
                            dist = float(cdist(emb_a, emb_b, metric='cosine')[0, 0])
                            outlier_distance_matrix[label_a][label_b] = round(dist, 4)
                console.print(
                    f"[info]Outlier analysis: {len(active_outlier_details)} active outliers, "
                    f"{outlier_health.get('active_outliers', 0)} from health check[/]"
                )
            except Exception as e:
                console.print(f"[warning]Could not compute outlier centroid distances: {e}[/warning]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return render_template("outliers.jinja", {
            "active_page": "outliers",
            "outlier_health": outlier_health,
            "active_outlier_details": active_outlier_details,
            "outlier_distance_matrix": outlier_distance_matrix,
            "computed_at": outlier_health.get("computed_at", datetime.now().isoformat()),
            "request": request,
        })
    except Exception as e:
        console.print(f"[error]Error computing outlier health: {e}[/]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise HTTPException(status_code=500, detail=f"Failed to compute outlier health: {e}")

@router.get("/timeline", response_class=HTMLResponse)
async def timeline_html(request: Request):
    """Speaker activity timeline visualization."""
    labeler = _check_labeler()
    try:
        timeline = labeler.get_speaker_timeline()
        console.print(f"[info]Rendering timeline page: {timeline.get('total_segments', 0)} segments[/]")
        return render_template("timeline.jinja", {
            "active_page": "timeline",
            "timeline": timeline,
            "computed_at": timeline.get("computed_at", datetime.now().isoformat()),
            "request": request,
        })
    except Exception as e:
        console.print(f"[error]Error computing timeline: {e}[/]")
        raise HTTPException(status_code=500, detail=f"Failed to compute timeline: {e}")

# Audio streaming endpoint
@router.get("/segment-audio")
async def serve_segment_audio(request: Request, segment_id: str = Query(None, description="Segment ID to serve audio for")):
    """
    Serves segment audio file with Range support for progressive streaming.
    Searches SEGMENT_AUDIO_DIR for files matching the segment_id.
    """
    if not segment_id:
        raise HTTPException(status_code=400, detail="segment_id parameter is required")
    
    console.print(f"[info]Looking for audio file for segment_id: {segment_id}[/info]")
    
    # Search for audio files matching the segment_id
    audio_files = get_audio_files()
    matching_files = [f for f in audio_files if segment_id in f['name'] or segment_id in f['path']]
    
    if not matching_files:
        console.print(f"[warning]No audio file found for segment_id: {segment_id}[/warning]")
        raise HTTPException(status_code=404, detail=f"No audio file found for segment {segment_id}")
    
    # Use the first matching file
    audio_file = matching_files[0]
    audio_file_path = audio_file['full_path']
    
    file_size = os.path.getsize(audio_file_path)
    range_header = request.headers.get("Range")
    
    console.print(f"[info]GET /segment-audio - File: {Path(audio_file_path).name} - Range: {range_header!r} from {request.client.host}[/info]")
    
    start, end = get_range_info(range_header, file_size)
    content_length = end - start + 1
    
    ext = Path(audio_file_path).suffix.lower()
    content_type_map = {
        '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.ogg': 'audio/ogg',
        '.flac': 'audio/flac', '.aac': 'audio/aac', '.m4a': 'audio/mp4',
        '.wma': 'audio/x-ms-wma'
    }
    content_type = content_type_map.get(ext, 'audio/mpeg')
    
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": content_type,
        "Cache-Control": "no-cache",
    }
    
    if range_header:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        status_code = 206
        console.print(f"[info]206 Partial - bytes {start}-{end}/{file_size} ({content_length} bytes)[/info]")
    else:
        status_code = 200
        console.print(f"[info]200 Full - serving entire file ({file_size} bytes)[/info]")
    
    generator = generate_audio_chunks(audio_file_path, start, end)
    return StreamingResponse(
        generator,
        status_code=status_code,
        headers=headers,
        media_type=content_type,
    )


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
