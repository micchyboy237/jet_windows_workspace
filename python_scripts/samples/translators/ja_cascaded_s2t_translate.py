from transformers import pipeline

# Short audio
AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

# Long audio
# AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

# load model
pipe = pipeline(
    model="japanese-asr/ja-cascaded-s2t-translation",
    model_kwargs={"attn_implementation": "sdpa"},
    model_translation="facebook/nllb-200-distilled-600M",
    tgt_lang="eng_Latn",
    chunk_length_s=15,
    trust_remote_code=True,
)

# Transcribe and translate
output = pipe(AUDIO_PATH)
print(f"Results: {output}")
