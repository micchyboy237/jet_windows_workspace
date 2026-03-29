"""
Usage Examples:
    # Auto-detect method (soft if available, else OCR)
    python extract-subtitles.py video.mp4
    # Save both .srt and segments.json into a custom directory
    python extract-subtitles.py video.mp4 -o my_results --method ocr
    # Process only from 2:30 to 10:00, stricter bottom filtering
    python extract-subtitles.py video.mp4 --start 150 --end 600 --bottom-threshold 0.70
    # Show help
    python extract-subtitles.py --help
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
from paddleocr import PaddleOCR
from rich import print
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


def seconds_to_srt_time(sec: float) -> str:
    hours = int(sec // 3600)
    minutes = int((sec // 60) % 60)
    seconds = int(sec % 60)
    milliseconds = int((sec % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def has_subtitle_tracks(video_path: Union[str, Path]) -> bool:
    video_path = str(Path(video_path).expanduser().resolve())
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-select_streams",
                "s",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        streams = result.stdout.strip().split("\n")
        return any(stream == "subtitle" for stream in streams)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def extract_soft_subtitles(
    video_path: Union[str, Path],
    output_srt: Union[str, Path],
    subtitle_index: int = 0,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> None:
    video_path = str(Path(video_path).resolve())
    output_srt = Path(output_srt).resolve()
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-map",
        f"0:s:{subtitle_index}",
        str(output_srt),
    ]

    if start is not None:
        cmd.extend(["-ss", str(start)])
    if end is not None:
        if start is not None:
            cmd.extend(["-t", str(end - start)])
        else:
            cmd.extend(["-to", str(end)])

    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install FFmpeg.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract soft subtitles: {e}")


def get_bottom_middle_texts(
    ocr_result: Any,
    img_height: int,
    img_width: int,
    bottom_threshold: float = 0.65,
    center_tolerance: float = 0.30,
) -> str:
    if not ocr_result:
        return ""

    if isinstance(ocr_result, list) and len(ocr_result) > 0:
        res = ocr_result[0]
    else:
        res = ocr_result

    if not isinstance(res, dict):
        return ""

    rec_texts = res.get("rec_texts", [])
    rec_scores = res.get("rec_scores", [])
    rec_polys = res.get("rec_polys", res.get("rec_boxes", []))

    if not rec_texts:
        return ""

    filtered = []
    bottom_y_cutoff = img_height * bottom_threshold
    left_limit = img_width * (0.5 - center_tolerance)
    right_limit = img_width * (0.5 + center_tolerance)

    for txt, score, poly in zip(rec_texts, rec_scores, rec_polys):
        if score > 0.5:
            avg_y = sum(p[1] for p in poly) / len(poly)
            xs = [p[0] for p in poly]
            min_x, max_x = min(xs), max(xs)
            center_x = (min_x + max_x) / 2

            if avg_y >= bottom_y_cutoff and left_limit <= center_x <= right_limit:
                filtered.append((txt, avg_y))

    filtered.sort(key=lambda x: x[1])
    return "\n".join(txt for txt, _ in filtered)


def extract_with_ocr(
    video_path: Union[str, Path],
    output_srt: Union[str, Path],
    interval: float = 0.5,
    crop_ratio: float = 0.33,
    lang: str = "en",
    start: Optional[float] = None,
    end: Optional[float] = None,
    bottom_threshold: float = 0.65,
    center_tolerance: float = 0.30,
) -> None:
    video_path = Path(video_path).resolve()
    output_srt = Path(output_srt).resolve()
    output_srt.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    effective_start = max(0.0, start) if start is not None else 0.0
    effective_end = min(duration_sec, end) if end is not None else duration_sec

    if effective_start >= effective_end:
        raise ValueError(
            f"Invalid time range: start ({effective_start}s) >= end ({effective_end}s) "
            f"(video duration: {duration_sec:.1f}s)"
        )

    effective_duration = effective_end - effective_start

    ocr = PaddleOCR(
        lang=lang,
        device="gpu",
        use_textline_orientation=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    previous_text = ""
    segments: List[Dict[str, Union[int, float, str]]] = []
    start_time: Optional[float] = None
    subtitle_index = 1
    steps = int(effective_duration / interval) + 1

    with (
        open(output_srt, "w", encoding="utf-8") as f,
        tqdm(total=steps, desc="OCR processing") as pbar,
    ):
        t = effective_start
        while t < effective_end:
            frame_idx = min(total_frames - 1, int(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            img = frame
            h, w = img.shape[:2]
            if crop_ratio > 0:
                img = img[int(h * (1 - crop_ratio)) :, :]
                h = img.shape[0]  # updated height after crop

            result = ocr.predict(img)
            current_text = get_bottom_middle_texts(
                result, h, w, bottom_threshold, center_tolerance
            )

            if current_text != previous_text:
                if previous_text and start_time is not None:
                    end_display = t
                    f.write(
                        f"{subtitle_index}\n"
                        f"{seconds_to_srt_time(start_time)} --> {seconds_to_srt_time(end_display)}\n"
                        f"{previous_text}\n\n"
                    )
                    segments.append(
                        {
                            "segment_num": subtitle_index,
                            "start": start_time,
                            "end": end_display,
                            "text": previous_text,
                        }
                    )
                    subtitle_index += 1
                start_time = t if current_text else None

            previous_text = current_text
            t += interval
            pbar.update(1)

        if previous_text and start_time is not None:
            f.write(
                f"{subtitle_index}\n"
                f"{seconds_to_srt_time(start_time)} --> {seconds_to_srt_time(effective_end)}\n"
                f"{previous_text}\n\n"
            )
            segments.append(
                {
                    "segment_num": subtitle_index,
                    "start": start_time,
                    "end": effective_end,
                    "text": previous_text,
                }
            )

    # Save segments.json in the same directory as .srt
    json_path = output_srt.with_name("segments.json")
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(segments, f_json, ensure_ascii=False, indent=2)

    cap.release()


def extract_subtitles(
    video_path: Union[str, Path],
    output_srt: Union[str, Path],
    method: str = "auto",
    interval: float = 0.5,
    crop_ratio: float = 0.33,
    start: Optional[float] = None,
    end: Optional[float] = None,
    lang: str = "en",
    bottom_threshold: float = 0.65,
    center_tolerance: float = 0.30,
    subtitle_index: int = 0,
) -> None:
    if method == "auto":
        method = "soft" if has_subtitle_tracks(video_path) else "ocr"

    if method == "soft":
        extract_soft_subtitles(
            video_path, output_srt, subtitle_index, start=start, end=end
        )
    elif method == "ocr":
        extract_with_ocr(
            video_path,
            output_srt,
            interval,
            crop_ratio,
            lang,
            start,
            end,
            bottom_threshold,
            center_tolerance,
        )
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract .srt subtitles from a video using soft track or OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    DEFAULT_VIDEO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\video\generated\extract_video_segment\video_segment.mp4"
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        default=DEFAULT_VIDEO_PATH,
        help=f"Path to the input video file (default: {DEFAULT_VIDEO_PATH})"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "ocr", "soft"],
        default="auto",
        help="Extraction method: auto (default), ocr, or soft",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval in seconds for OCR (default: 0.5)",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.33,
        help="Bottom crop ratio for OCR focus (0 to disable, default: 0.33)",
    )
    parser.add_argument(
        "--lang", type=str, default="en", help="Language code for OCR (default: en)"
    )
    parser.add_argument(
        "--start", type=float, default=None, help="Start time in seconds (optional)"
    )
    parser.add_argument(
        "--end", type=float, default=None, help="End time in seconds (optional)"
    )
    parser.add_argument(
        "--bottom-threshold",
        type=float,
        default=0.65,
        help="Y-threshold (0-1) below which text is considered bottom (default: 0.65)",
    )
    parser.add_argument(
        "--center-tolerance",
        type=float,
        default=0.30,
        help="Horizontal center tolerance (0-0.5); text center must be within ±tolerance from 0.5 (default: 0.30)",
    )
    parser.add_argument(
        "--subtitle-index",
        type=int,
        default=0,
        help="Subtitle track index for soft extraction (default: 0)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    shutil.rmtree(output_dir, ignore_errors=True)

    srt_path = output_dir / "subtitles.srt"
    segments_path = srt_path.with_name("segments.json")

    try:
        extract_subtitles(
            args.video,
            srt_path,
            method=args.method,
            interval=args.interval,
            crop_ratio=args.crop_ratio,
            start=args.start,
            end=args.end,
            lang=args.lang,
            bottom_threshold=args.bottom_threshold,
            center_tolerance=args.center_tolerance,
            subtitle_index=args.subtitle_index,
        )

        print(f"[green]Subtitles extracted to {srt_path}[/green]")

        if args.method in ("auto", "ocr"):
            print(f"[green]Segments saved to {segments_path}[/green]")

    except Exception as e:
        print(f"[red]Error: {str(e)}[/red]")
        exit(1)
