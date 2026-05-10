"""
Usage Examples:

    # Extract frames at specific timestamps (in seconds)
    python extract-video-frames.py video.mp4 --at-seconds 1.5 5.0 12.3 --output-dir shots

    # Extract a frame every 3 seconds, limit to 30 frames, lower quality
    python extract-video-frames.py input.mov --every-seconds 3 --max-frames 30 --quality 80 -o keyframes

    # Extract specific frame numbers
    python extract-video-frames.py clip.mp4 --at-frames 0 100 250 500 1200 --output-dir frames

    # Extract one frame every 200 frames
    python extract-video-frames.py movie.mkv --every-frames 200

    # Just save the very first frame
    python extract-video-frames.py video.mp4 --first --output-dir thumbs

    # Extract frames at 10%, 30%, 50%, 70%, 90% of the video (manual calculation)
    python extract-video-frames.py demo.mp4 --at-seconds 12.4 37.2 62.0 86.8 --output-dir percent

    # Show help
    python extract-video-frames.py --help

Note: Exactly one of these modes is required:
  --at-seconds    --at-frames    --every-seconds    --every-frames    --first
"""

import cv2
import os
from pathlib import Path
from typing import Union, List, Optional
import argparse


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    timestamps_sec: Optional[List[float]] = None,
    frame_numbers: Optional[List[int]] = None,
    every_n_seconds: Optional[float] = None,
    every_n_frames: Optional[int] = None,
    max_frames: int = 100,
    jpg_quality: int = 92
) -> List[str]:
    """
    Extract one or more frames from a video file as JPEG images.
    """
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    saved_paths = []
    count = 0

    try:
        if timestamps_sec:
            for t in sorted(timestamps_sec):
                if t < 0 or t > duration_sec:
                    continue
                frame_idx = min(total_frames - 1, int(round(t * fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                out_path = output_dir / f"frame_{t:06.2f}s_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                saved_paths.append(str(out_path))
                count += 1
                if count >= max_frames:
                    break

        elif frame_numbers:
            for idx in sorted(set(frame_numbers)):  # remove duplicates
                if idx < 0 or idx >= total_frames:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                out_path = output_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                saved_paths.append(str(out_path))
                count += 1
                if count >= max_frames:
                    break

        elif every_n_seconds is not None and every_n_seconds > 0:
            step = max(1, int(round(every_n_seconds * fps)))
            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break

                t = i / fps
                out_path = output_dir / f"frame_{t:06.2f}s_{i:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                saved_paths.append(str(out_path))
                count += 1
                if count >= max_frames:
                    break

        elif every_n_frames is not None and every_n_frames > 0:
            step = max(1, every_n_frames)
            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break

                out_path = output_dir / f"frame_{i:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                saved_paths.append(str(out_path))
                count += 1
                if count >= max_frames:
                    break

        else:
            # Default / --first
            ret, frame = cap.read()
            if ret:
                out_path = output_dir / "frame_000000.jpg"
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                saved_paths.append(str(out_path))

    finally:
        cap.release()

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file as JPEG images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exactly one extraction mode is required."
    )

    parser.add_argument("video", type=str, help="Path to the input video file")

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="frames",
        help="Directory to save extracted frames (default: ./frames)"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--at-seconds",
        type=float,
        nargs="+",
        metavar="SEC",
        help="Extract frames at these exact times (seconds)"
    )

    group.add_argument(
        "--at-frames",
        type=int,
        nargs="+",
        metavar="N",
        help="Extract frames at these exact frame numbers"
    )

    group.add_argument(
        "--every-seconds",
        type=float,
        metavar="INTERVAL",
        help="Extract a frame every X seconds"
    )

    group.add_argument(
        "--every-frames",
        type=int,
        metavar="STEP",
        help="Extract a frame every X frames"
    )

    group.add_argument(
        "--first",
        action="store_true",
        help="Extract only the first frame"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to extract (default: 100)"
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=92,
        choices=range(10, 101),
        metavar="[10-100]",
        help="JPEG quality 10–100 (default: 92)"
    )

    args = parser.parse_args()

    try:
        saved = extract_frames(
            video_path=args.video,
            output_dir=args.output_dir,
            timestamps_sec=args.at_seconds,
            frame_numbers=args.at_frames,
            every_n_seconds=args.every_seconds,
            every_n_frames=args.every_frames,
            max_frames=args.max_frames,
            jpg_quality=args.quality
        )

        if saved:
            print(f"Successfully extracted {len(saved)} frame(s):")
            for path in saved:
                print(f"  • {path}")
        else:
            print("Warning: No frames were extracted.")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
