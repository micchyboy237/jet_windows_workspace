from transformers import pipeline

REPO_ID = "litagin/anime_speech_emotion_classification"
pipe = pipeline(
    "audio-classification",
    model=REPO_ID,
    feature_extractor=REPO_ID,
    trust_remote_code=True,
    device="cuda",
)

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\python_scripts\samples\audio\data\sound.wav"
result = pipe(audio_path)
print(result)
