import os
import shutil
import dataclasses
import torch
from faster_whisper import WhisperModel
from utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

audio_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data/audio/1.wav")
)

# Device detection
if torch.cuda.is_available():
    device = "cuda"
    print(f"[Device] Using CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = "cpu"
    print("[Device] Using CPU")

# UPDATED — replace previous WhisperModel init
model = WhisperModel("large-v3", device=device)


# Transcribe Japanese audio and translate to English
segments_iter, info = model.transcribe(
    audio=audio_path,
    language="ja",
    task="translate",

    # # Decoding: Maximum accuracy
    # beam_size=10,
    # patience=2.0,
    # temperature=0.0,
    # length_penalty=1.0,
    # best_of=1,
    # log_prob_threshold=-0.5,

    # # Context & consistency
    # condition_on_previous_text=True,

    # # Japanese punctuation handling
    # prepend_punctuations="\"'“¿([{-『「（［",
    # append_punctuations="\"'.。,，!！?？:：”)]}、。」」！？",

    # # Clean input
    # vad_filter=True,
    # vad_parameters=None,

    # Output options
    without_timestamps=False,
    word_timestamps=True,
    chunk_length=30,
    log_progress=True,
)

save_file(info, f"{OUTPUT_DIR}/1_wav/info.json")

print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
segments = []
for segment in segments_iter:
    segment_dict = dataclasses.asdict(segment)
    segments.append(segment_dict)
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

save_file(segments, f"{OUTPUT_DIR}/1_wav/segments.json")