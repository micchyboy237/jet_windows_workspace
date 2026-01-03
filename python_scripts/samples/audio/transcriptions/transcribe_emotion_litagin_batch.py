from transformers import pipeline
from utils import resolve_audio_paths

REPO_ID = "litagin/anime_speech_emotion_classification"
pipe = pipeline(
    "audio-classification",
    model=REPO_ID,
    feature_extractor=REPO_ID,
    trust_remote_code=True,
    device="cuda",  # or device=0 for GPU
    # Optional: control batch size explicitly (default is often 1, but pipeline auto-batches when list input is given)
    # batch_size=8,  # Adjust based on your GPU memory (e.g., 8-32 for GTX 1660)
)

audio_dir = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\servers\live_subtitles\generated\live_subtitles_server_for_test"
audio_paths = resolve_audio_paths(audio_dir, recursive=True)

results = pipe(audio_paths)  # Returns a list of lists (one list of dicts per audio)

for path, result in zip(audio_paths, results):
    print(f"\nResults for {path}:")
    print(result[:5])  # Top 5 labels, similar to your single example