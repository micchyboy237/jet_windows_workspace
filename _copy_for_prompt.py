import os
import fnmatch
import argparse
import subprocess
import json
import tiktoken
from rich.console import Console
from tqdm import tqdm
from _utils_copy_for_prompt import (
    find_files,
    format_file_structure,
    clean_newlines,
    clean_content,
    remove_parent_paths,
    copy_to_clipboard,
)
from headroom import compress

logger = Console()

exclude_files = [
    "**/.git/",
    "**/.gitignore",
    "**/.DS_Store",
    "**/*.pyc",
    "**/_copy*.py",
    "**/__pycache__/",
    "**/.pytest_cache/",
    "**/node_modules/",
    "**/*lock.json",
    "**/*.lock",
    "**/public/",
    "**/mocks/",
    "**/.venv/",
    "**/dream/",
    "**/jupyter/",
    "**/*.png",
    "**/*.svg",
    # "**/_*",
    # "**/.cache/",
    "**/_git_stats.json",
    "**/stats_results/",
    # "**/generated/",
    # "**/.*",

    # Custom
    # "**/*.sh"
    # "**/__init__.py",
    # "*.md",
]
include_files = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Examples\.vscode\launch.json",

    # r"C:\Users\druiv\Desktop\Jet_Files\Cloned_Repos\WhisperJAV\whisperjav\main.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\state.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_segment_speaker_labeler.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\dashboard.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\speaker_metrics.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\single_plot.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\health_diagnostics.js",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\independent_analysis.js",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\similarity_network.js",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\pairwise_comparison.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\speaker_embedding_plot.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\dimension_diff_view.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\similarity_gauge.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\speaker_embedding_plot.html",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\pairwise_comparison.js",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\similarity_network.js",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers\dimension_diff_view.js",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\components\dimension_diff_view.html",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_metrics_mixin.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\helpers\speaker_metrics.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\segment_speaker_labeler_health_mixin.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\speakers.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\test_segment_speaker_labeler_mixin_inheritance.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\vad_firered.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speech_waves.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_speech_waves.py",
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\temp\temp9.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\temp\temp10.py",
   #  r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\main\_main_segment_speaker_labeler.py",
    r"",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\main.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\config.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\audio_config.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers\segment_detail.html",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\speakers.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\state.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\core\processing.py",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\services\speaker_metrics_mixin.py",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\audio_streaming\demo1\player.html",
    r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\audio_streaming\demo1\main.py",
    r"",
]

structure_include = [
    r"",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\speakers",
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\static\js\speakers",
]
structure_exclude = []

include_content = [
    # r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\templates\tagger",
]
exclude_content = []

# Args defaults
SHORTEN_FUNCTS = False 
INCLUDE_FILE_STRUCTURE = False

COMPRESSION_MODEL = "gpt-4o"
TOKEN_BUDGET = 8000

DEFAULT_QUERY_MESSAGE = r"""
Analyze carefully why demo1 works and audio player in segment detail does not.

Now refactor segment_detail.html to move out all audio player logic on a separate audio_player.html. Use include.

Then also implement with howler.


# "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\live_subtitles_server2_with_en\routes\speakers.py

@router.get("/segment/{segment_id}/audio")
async def get_segment_audio(segment_id: str, request: Request):
    \"\"\"
    Get the audio data for a specific segment (for playback/download).
    Supports HTTP Range requests for proper streaming/seeking.
    \"\"\"
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
                        import tempfile
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
    
    # === Serve with Range request support ===
    file_size = audio_path.stat().st_size
    duration_sec = get_audio_duration(str(audio_path))
    
    # Parse Range header
    range_header = request.headers.get("range")
    
    if range_header:
        # Handle partial content request
        try:
            range_value = range_header.replace("bytes=", "")
            start_str, end_str = range_value.split("-")
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            
            if start >= file_size:
                raise HTTPException(
                    status_code=416,
                    detail="Range not satisfiable"
                )
            
            end = min(end, file_size - 1)
            chunk_size = end - start + 1
            
            console.print(
                f"[info]Serving audio range: bytes {start}-{end}/{file_size} "
                f"({chunk_size} bytes)[/]"
            )
            
            # Read only the requested chunk
            def file_iterator(file_path, start_byte, end_byte):
                with open(file_path, "rb") as f:
                    f.seek(start_byte)
                    remaining = end_byte - start_byte + 1
                    chunk_size = min(8192, remaining)
                    while remaining > 0:
                        data = f.read(min(chunk_size, remaining))
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
            }
            
            return StreamingResponse(
                file_iterator(audio_path, start, end),
                status_code=206,  # Partial Content
                headers=headers,
                media_type="audio/wav",
            )
            
        except (ValueError, IndexError) as e:
            console.print(f"[warning]Invalid Range header: {range_header} - {e}[/]")
            # Fall through to serve full file
    
    # Full file response with proper headers
    console.print(f"[info]Serving full audio file: {file_size} bytes[/]")
    
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
    
    # Use streaming for consistent behavior
    def full_file_iterator(file_path):
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk
    
    return StreamingResponse(
        full_file_iterator(audio_path),
        status_code=200,
        headers=headers,
        media_type="audio/wav",
    )


@router.get("/segment/{segment_id}", response_class=HTMLResponse)
async def get_segment_detail_page(request: Request, segment_id: str):
    \"\"\"
    Serve a detailed page for a specific segment with play/download audio buttons.
    \"\"\"
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
        f"audio_duration={audio_duration:.3f}s[/]"  # ✅ Log both
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

"""

DEFAULT_INSTRUCTIONS_MESSAGE = """
General:
- Browse when beneficial or requested.
- Keep explanations simple and clear.
When coding:
- Provide step-by-step analysis and explain the flow.
- Use visuals, diagrams, or tables when helpful.
- Show full code for new files, then show full function code for new or updated functions.
- Write smart, flexible, reusable, maintainable, optimal, robust, and minimal code.
- Always add logs so we can trace and know if all features work correctly.
""".strip()

DEFAULT_SYSTEM_MESSAGE = """
""".strip()
# For existing projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
# "\n- Only respond with parts of the code that have been added or updated to keep it short and concise."
# )z
# For creating projects
# DEFAULT_INSTRUCTIONS_MESSAGE += (
# "\n- At the end, display the updated file structure and instructions for running the code."
# "\n- Provide complete working code for each file (should match file structure)"
# )
# base_dir should be actual file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(file_dir)


def get_language_from_extension(filename: str) -> str:
    """
    Simple file extension → markdown code fence language mapping
    Returns 'text' as safe fallback
    """
    ext = os.path.splitext(filename.lower())[1]
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".md": "markdown",
        ".mdx": "mdx",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".sh": "bash",
        ".bash": "bash",
        ".sql": "sql",
        ".prisma": "prisma",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".php": "php",
        ".rb": "ruby",
    }
    return mapping.get(ext, "text")


def main():
    global exclude_files, include_files, include_content, exclude_content
    print("Running _copy_for_prompt.py")
    # Parse command-line options
    parser = argparse.ArgumentParser(
        description="Generate clipboard content from specified files."
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        default=file_dir,
        help="Base directory to search files in (default: current directory)",
    )
    parser.add_argument(
        "-if",
        "--include-files",
        nargs="*",
        default=include_files,
        help="Patterns of files to include (default: schema.prisma, episode)",
    )
    parser.add_argument(
        "-ef",
        "--exclude-files",
        nargs="*",
        default=exclude_files,
        help="Directories or files to exclude (default: node_modules)",
    )
    parser.add_argument(
        "-ic",
        "--include-content",
        nargs="*",
        default=include_content,
        help="Patterns of file content to include",
    )
    parser.add_argument(
        "-ec",
        "--exclude-content",
        nargs="*",
        default=exclude_content,
        help="Patterns of file content to exclude",
    )
    parser.add_argument(
        "-cs",
        "--case-sensitive",
        action="store_true",
        default=False,
        help="Make content pattern matching case-sensitive",
    )
    parser.add_argument(
        "-sf",
        "--shorten-funcs",
        action="store_true",
        default=SHORTEN_FUNCTS,
        help="Shorten function and class definitions",
    )
    parser.add_argument(
        "-s",
        "--system",
        default=DEFAULT_SYSTEM_MESSAGE,
        help="Message to include in the clipboard content",
    )
    parser.add_argument(
        "-m",
        "--message",
        default=DEFAULT_QUERY_MESSAGE,
        help="Message to include in the clipboard content",
    )
    parser.add_argument(
        "-i",
        "--instructions",
        default=DEFAULT_INSTRUCTIONS_MESSAGE,
        help="Instructions to include in the clipboard content",
    )
    parser.add_argument(
        "-fo",
        "--filenames-only",
        action="store_true",
        help="Only copy the relative filenames, not their contents",
    )
    parser.add_argument(
        "-nl",
        "--no-length",
        action="store_true",
        default=INCLUDE_FILE_STRUCTURE,
        help="Do not show file character length",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        default=False,
        help="Enable compression of the clipboard content before copying (default: False)",
    )
    args = parser.parse_args()
    base_dir = args.base_dir
    include = args.include_files
    exclude = args.exclude_files
    include_content = args.include_content
    exclude_content = args.exclude_content
    case_sensitive = args.case_sensitive
    shorten_funcs = args.shorten_funcs
    query_message = args.message
    system_message = args.system
    instructions_message = args.instructions
    filenames_only = args.filenames_only
    show_file_length = not args.no_length
    compress_enabled = args.compress
    # Find all files matching the patterns in the base directory and its subdirectories
    print("\n")
    context_files = find_files(
        base_dir, include, exclude, include_content, exclude_content, case_sensitive
    )
    print("\n")
    print(f"Include patterns: {include}")
    print(f"Exclude patterns: {exclude}")
    print(f"Include content patterns: {include_content}")
    print(f"Exclude content patterns: {exclude_content}")
    print(f"Case sensitive: {case_sensitive}")
    print(f"Filenames only: {filenames_only}")
    print(f"Compress enabled: {compress_enabled}")
    print(
        f"\nFound files ({len(context_files)}):\n{json.dumps(context_files, indent=2)}"
    )
    print("\n")
    # Initialize the clipboard content
    clipboard_content = ""
    if not context_files:
        print("No context files found matching the given patterns.")
    else:
        # Append relative filenames to the clipboard content
        for file in tqdm(
            context_files, desc=f"Processing {len(context_files)} files..."
        ):
            rel_path = os.path.relpath(path=file, start=file_dir)
            cleaned_rel_path = remove_parent_paths(rel_path)
            prefix = f"\n# {cleaned_rel_path}\n" if not filenames_only else f"{file}\n"
            if filenames_only:
                clipboard_content += f"{prefix}"
            else:
                file_path = os.path.relpath(os.path.join(base_dir, file))
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                            content = clean_content(content, file, shorten_funcs)
                            # ── NEW: Add fenced code block ───────────────────────────────
                            lang = get_language_from_extension(file)
                            fenced_content = f"```{lang}\n{content.rstrip()}\n```"
                            clipboard_content += f"{prefix}{fenced_content}\n\n"
                    except Exception:
                        # Continue to the next file
                        continue
                else:
                    clipboard_content += f"{prefix}\n"
        clipboard_content = clean_newlines(clipboard_content).strip()
    # Generate and format the file structure
    structure_include_files = structure_include
    if include:
        structure_include_files += include
    structure_exclude_files = structure_exclude
    if exclude:
        structure_exclude_files += exclude
    files_structure = format_file_structure(
        base_dir,
        include_files=structure_include_files,
        exclude_files=structure_exclude_files,
        include_content=include_content,
        exclude_content=exclude_content,
        case_sensitive=case_sensitive,
        shorten_funcs=shorten_funcs,
        show_file_length=show_file_length,
    )
    # Prepend system and query to the clipboard content then append instructions
    clipboard_content_parts = []
    if system_message:
        clipboard_content_parts.append(f"System\n{system_message}\n")
    # Query should come before instructions
    clipboard_content_parts.append(f"{query_message}\n\n")
    if instructions_message:
        clipboard_content_parts.append(f"Instructions\n{instructions_message}\n")
    if INCLUDE_FILE_STRUCTURE:
        clipboard_content_parts.append(f"Files Structure\n{files_structure}\n")
    if clipboard_content:
        clipboard_content_parts.append(
            f"Existing Files Contents\n{clipboard_content}\n"
        )
    clipboard_content = "\n\n".join(clipboard_content_parts)
    # Compress to reduce tokens (optional)
    if compress_enabled:
        messages = [{"role": "user", "content": clipboard_content}]
        result = compress(
            messages,
            model=COMPRESSION_MODEL,  # headroom uses this for strategy selection only
            token_budget=TOKEN_BUDGET,  # enforce fit within llama-server context
            ccr_enabled=True,  # reversible compression (default)
            compress_user_messages=True,
            target_ratio=0.5,  # keep 50% — safe for mixed prose + code
            protect_recent=0,  # only 1 message, nothing to protect
            protect_analysis_context=False,  # do not protect code from compression
            # kompress_model="disabled",
        )
        # Log compression stats using logger.log for each result.*
        logger.log("Tokens before:", f"{result.tokens_before:,}")
        logger.log("Tokens after:", f"{result.tokens_after:,}")
        logger.log(
            "Tokens saved:",
            f"{result.tokens_saved:,} ({result.compression_ratio:.1%})",
        )
        logger.log(
            "Transforms applied:",
            str(result.transforms_applied),
        )
    else:
        logger.log("Compression skipped (use -c or --compress to enable)")
    # Copy the content to the clipboard
    copy_to_clipboard(clipboard_content)
    # Print the copied content character count
    logger.log("Prompt Char Count:", len(clipboard_content))
    logger.log("Tokens Count (gpt-4o):", count_tokens(clipboard_content))
    # Newline
    print("\n")


def count_tokens(
    text: str,
    model: str = "gpt-4o",  # Best default
    encoding_name: str | None = None,
) -> int:
    """
    Count the number of tokens in a string using tiktoken.
    Args:
        text: The input string to tokenize.
        model: OpenAI model name to determine the encoding
               (default: "gpt-4o" — recommended).
        encoding_name: Optional direct encoding name
                       (e.g., "o200k_base", "cl100k_base").
                       Takes precedence over model.
    Returns:
        Number of tokens.
    """
    if encoding_name:
        encoding = tiktoken.get_encoding(encoding_name)
    else:
        encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


if __name__ == "__main__":
    main()
