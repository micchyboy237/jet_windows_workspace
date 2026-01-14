import torch
import librosa
import numpy as np
from transformers import AutoProcessor, Wav2Vec2ForCTC

# Short audio
AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

# Long audio
# AUDIO_PATH = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"

model = Wav2Vec2ForCTC.from_pretrained(
    "reazon-research/japanese-wav2vec2-large-rs35kh",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
).to("cuda")
processor = AutoProcessor.from_pretrained("reazon-research/japanese-wav2vec2-large-rs35kh")

audio, _ = librosa.load(AUDIO_PATH, sr=16_000)
audio = np.pad(audio, pad_width=int(0.5 * 16_000))  # Recommend to pad audio before inference
input_values = processor(
    audio,
    return_tensors="pt",
    sampling_rate=16_000
).input_values.to("cuda").to(torch.bfloat16)

with torch.inference_mode():
    logits = model(input_values).logits.cpu()
predicted_ids = torch.argmax(logits, dim=-1)[0]
transcription = processor.decode(predicted_ids, skip_special_tokens=True)

print(transcription)