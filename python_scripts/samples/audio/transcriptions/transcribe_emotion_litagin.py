from transformers import pipeline

REPO_ID = "litagin/anime_speech_emotion_classification"
pipe = pipeline(
    "audio-classification",
    model=REPO_ID,
    feature_extractor=REPO_ID,
    trust_remote_code=True,
    device="cuda",
)

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
result = pipe(audio_path)
print(result)
