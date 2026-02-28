from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from translators.translate_jp_en_lfm2 import translate_text

# Load the model (automatically downloads from HF or ModelScope)
model_dir = "FunAudioLLM/SenseVoiceSmall"   # or "iic/SenseVoiceSmall"

audio_path = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"

model = AutoModel(
    model=model_dir,
    disable_update=True,
    vad_model="fsmn-vad",      # optional: voice activity detection
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",           # or "cpu"
    hub="hf",                   # or "ms" for ModelScope
    # trust_remote_code=True,
    # remote_code="./model.py",  # usually not needed if using latest FunASR
)

# Generate transcription + emotion + events
res = model.generate(
    input=audio_path,   # or local path: "audio.mp3", "audio.wav", etc.
    cache={},
    language="ja",           # "auto" / "zh" / "en" / "yue" / "ja" / "ko" / "nospeech"
    use_itn=False,              # inverse text normalization (e.g. numbers → digits)
    batch_size=32,            # adjust based on GPU memory
    output_timestamp=True,
    merge_vad=True,
    merge_length_s=15,
)

# Example output structure:
# [{'text': '<|en|><|NEUTRAL|> Hello this is a test.', 'timestamp': ..., 'emo': 'NEUTRAL', ...}]



ja_text = rich_transcription_postprocess(res[0]["text"])
en_text = translate_text(ja_text)

print("\n")
print(f"JA: {ja_text}")
print(f"EN: {en_text}")
# Output: どうしたっけ？食べにくいじゃん。今日ね。担任の先生から電話があった。現地。