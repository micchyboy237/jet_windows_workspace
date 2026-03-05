import shutil
import json
from pathlib import Path
import cv2
from tqdm import tqdm
from paddleocr import PaddleOCR

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_file = "~/.cache/video/0001_video_en_sub.mp4"
video_file = str(Path(video_file).expanduser().resolve())

OCR_EVERY_N_SECONDS = 1.5          # OCR interval
MIN_TEXT_CHANGE_LEN = 3            # ignore very short noisy changes
MIN_SEGMENT_DURATION = 0.8         # seconds

# ────────────────────────────────────────────────
# Initialize
# ────────────────────────────────────────────────

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # You can also try: ocr_version="PP-OCRv4" or "PP-OCRv5"
    # det_db_box_thresh=0.4,   # sometimes helps with burned-in subs
)

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Cannot open video:", video_file)
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video:   {Path(video_file).name}")
print(f"FPS:     {fps:.2f}")
print(f"Frames:  {total_frames:,d}")
print(f"OCR every ~{OCR_EVERY_N_SECONDS:.1f} s  →  every {int(fps * OCR_EVERY_N_SECONDS)} frames\n")

# ────────────────────────────────────────────────
# Main loop with progress bar
# ────────────────────────────────────────────────

segments = []
prev_text = ""
start_time = None
segment_num = 1

# We use total_frames for nice tqdm bar
pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

frame_num = 0
ocr_interval_frames = int(fps * OCR_EVERY_N_SECONDS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pbar.update(1)
    frame_num += 1

    # Only run OCR every N seconds
    if frame_num % ocr_interval_frames != 0:
        continue

    # bottom ~25% — typical subtitle location
    h, w = frame.shape[:2]
    crop = frame[int(h * 0.74):, :, :]   # ← you can tune 0.74–0.80

    # ── You asked about predict_iter ─────────────────────────────
    # 
    #    result = list(ocr.predict_iter(crop))     # ← works, but usually overkill for single image
    #
    # Most people just do:
    result = ocr.predict(crop)           # returns list of list of [box, (text, score)]

    if not result or not result[0]:
        current_text = ""
    else:
        current_text = " ".join(line[1][0] for line in result[0]).strip()

    # Skip almost-empty noisy detections
    if len(current_text) < MIN_TEXT_CHANGE_LEN:
        current_text = ""

    if current_text != prev_text:
        # Close previous segment if exists
        if start_time is not None and prev_text:
            end_sec = frame_num / fps
            duration = end_sec - start_time

            if duration >= MIN_SEGMENT_DURATION:
                segments.append({
                    "segment_num": segment_num,
                    "start": start_time,
                    "end": end_sec,
                    "text": prev_text,
                })
                segment_num += 1

        # Start new segment
        if current_text:
            start_time = frame_num / fps
        else:
            start_time = None

        prev_text = current_text

# Don't forget the last segment
if start_time is not None and prev_text:
    end_sec = frame_num / fps
    duration = end_sec - start_time
    if duration >= MIN_SEGMENT_DURATION:
        segments.append({
            "segment_num": segment_num,
            "start": start_time,
            "end": end_sec,
            "text": prev_text,
        })

cap.release()
pbar.close()

# ────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────

json_path = OUTPUT_DIR / "ocr_segments.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(segments, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(segments)} segments → {json_path.resolve()}\n")

# Optional: quick look
for seg in segments[:6]:
    print(f"{seg['segment_num']:2d} | {seg['start']:6.1f} → {seg['end']:6.1f} | {seg['text']}")
if len(segments) > 6:
    print("...")
