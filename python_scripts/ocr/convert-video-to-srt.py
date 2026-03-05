"""
Usage examples:

    # Process full video with default settings
    python convert-video-to-srt.py

    # Process a specific time range
    python convert-video-to-srt.py video.mp4 --start 60 --end 180

    # More frequent OCR, stricter text change detection
    python convert-video-to-srt.py myvideo.mkv --ocr-every 1.0 --min-change 6 --min-duration 1.2

    # Custom crop (lower value = larger subtitle area)
    python convert-video-to-srt.py --crop-bottom 0.70 --start 300

    # Custom output folder name
    python convert-video-to-srt.py video.mp4 --output-dir-name "episode-03-subs"

    # Show help
    python convert-video-to-srt.py --help
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
from paddleocr import PaddleOCR
from tqdm import tqdm

# ────────────────────────────────────────────────
# Argument Parsing
# ────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Extract subtitle-like segments from video using PaddleOCR (bottom crop)"
)

parser.add_argument(
    "video",
    type=str,
    nargs="?",
    default="~/.cache/video/0001_video_en_sub.mp4",
    help="Path to the input video file (default: ~/.cache/video/0001_video_en_sub.mp4)"
)

parser.add_argument(
    "--start",
    type=float,
    default=None,
    help="Start time in seconds (optional)"
)

parser.add_argument(
    "--end",
    type=float,
    default=None,
    help="End time in seconds (optional)"
)

parser.add_argument(
    "--ocr-every",
    type=float,
    default=1.5,
    help="Perform OCR every X seconds (default: 1.5)"
)

parser.add_argument(
    "--min-change",
    type=int,
    default=3,
    help="Minimum length of text to consider it a real change (default: 3)"
)

parser.add_argument(
    "--min-duration",
    type=float,
    default=0.8,
    help="Minimum segment duration in seconds (default: 0.8)"
)

parser.add_argument(
    "--crop-bottom",
    type=float,
    default=0.74,
    help="Crop from this fraction of height from the top (default: 0.74 → bottom 26%%)"
)

parser.add_argument(
    "--output-dir-name",
    type=str,
    default=None,
    help="Custom name for output folder (default: script name)"
)

args = parser.parse_args()

# ────────────────────────────────────────────────
# Prepare paths & output
# ────────────────────────────────────────────────

video_path = Path(args.video).expanduser().resolve()
if not video_path.is_file():
    print(f"Error: Video file not found\n  {video_path}")
    exit(1)

if args.output_dir_name:
    output_dir = Path(__file__).parent / "generated" / args.output_dir_name
else:
    output_dir = Path(__file__).parent / "generated" / Path(__file__).stem

shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────
# Initialize OCR
# ────────────────────────────────────────────────

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # Optional tuning examples (uncomment if needed):
    # ocr_version="PP-OCRv5",
    # det_db_box_thresh=0.4,
    # det_db_thresh=0.3,
)

# ────────────────────────────────────────────────
# Video setup
# ────────────────────────────────────────────────

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Cannot open video:", video_path)
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps if fps > 0 else 0

print(f"Video:      {video_path.name}")
print(f"FPS:        {fps:.2f}")
print(f"Duration:   {video_duration:.1f} s")
print(f"Frames:     {total_frames:,d}")

# Handle start / end times
start_sec = max(0.0, args.start if args.start is not None else 0.0)
end_sec   = args.end if args.end is not None else video_duration
end_sec   = min(end_sec, video_duration)

if start_sec >= end_sec:
    print("Error: start time must be before end time")
    cap.release()
    exit(1)

start_frame = int(start_sec * fps)
end_frame   = min(total_frames, int(end_sec * fps) + 1)   # inclusive

ocr_interval_frames = max(1, int(fps * args.ocr_every))

print(f"Processing: {start_sec:.1f} → {end_sec:.1f} s "
      f"({start_frame:,d} – {end_frame:,d} frames)")
print(f"OCR every ~{args.ocr_every:.1f} s → every {ocr_interval_frames} frames")
print(f"Min text change len: {args.min_change}")
print(f"Min segment duration: {args.min_duration:.1f} s\n")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# ────────────────────────────────────────────────
# Main processing loop
# ────────────────────────────────────────────────

segments = []
prev_text = ""
start_time = None
segment_num = 1

pbar = tqdm(
    total=(end_frame - start_frame),
    desc="Processing video",
    unit="frame"
)

frame_num = start_frame

while cap.isOpened() and frame_num < end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    pbar.update(1)
    frame_num += 1

    if (frame_num - start_frame) % ocr_interval_frames != 0:
        continue

    # Crop bottom part (subtitles usually appear here)
    h, w = frame.shape[:2]
    crop_top = int(h * args.crop_bottom)
    crop = frame[crop_top:, :, :]

    result = ocr.predict(crop)

    if not result or not result[0]:
        current_text = ""
    else:
        current_text = " ".join(line[1][0] for line in result[0]).strip()

    if len(current_text) < args.min_change:
        current_text = ""

    current_time = frame_num / fps

    if current_text != prev_text:
        # Close previous segment
        if start_time is not None and prev_text:
            duration = current_time - start_time
            if duration >= args.min_duration:
                segments.append({
                    "segment_num": segment_num,
                    "start": start_time,
                    "end": current_time,
                    "text": prev_text,
                })
                segment_num += 1

        # Open new segment if there's text
        start_time = current_time if current_text else None
        prev_text = current_text

# Last segment
if start_time is not None and prev_text:
    current_time = min(frame_num / fps, end_sec)
    duration = current_time - start_time
    if duration >= args.min_duration:
        segments.append({
            "segment_num": segment_num,
            "start": start_time,
            "end": current_time,
            "text": prev_text,
        })

cap.release()
pbar.close()

# ────────────────────────────────────────────────
# Save results
# ────────────────────────────────────────────────

json_path = output_dir / "ocr_segments.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(segments, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(segments)} segments → {json_path.resolve()}\n")

# Preview first few segments
for seg in segments[:6]:
    print(f"{seg['segment_num']:2d} | {seg['start']:7.1f} → {seg['end']:7.1f} | {seg['text']}")
if len(segments) > 6:
    print("...")
