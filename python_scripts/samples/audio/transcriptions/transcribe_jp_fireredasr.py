from pathlib import Path
from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig

cache_dir = Path("~/.cache/pretrained_models").expanduser().resolve()
asr_model_dir = cache_dir / "FireRedASR2-AED"
vad_model_dir = cache_dir / "FireRedVAD/VAD"
lid_model_dir = cache_dir / "FireRedLID"
punc_model_dir = cache_dir / "FireRedPunc"

asr_system_config = FireRedAsr2SystemConfig(
    asr_model_dir=asr_model_dir,
    vad_model_dir=vad_model_dir,
    lid_model_dir=lid_model_dir,
    punc_model_dir=punc_model_dir,
    enable_lid=False,
    enable_punc=False,
)  # Use default config
asr_system = FireRedAsr2System(asr_system_config)

result = asr_system.process("C:/Users/druiv/Desktop/Jet_Files/Cloned_Repos/FireRedASR2S/assets/hello_zh.wav")
print(result)
# {'uttid': 'tmpid', 'text': '你好世界。', 'sentences': [{'start_ms': 440, 'end_ms': 1820, 'text': '你好世界。', 'asr_confidence': 0.868, 'lang': 'zh mandarin', 'lang_confidence': 0.999}], 'vad_segments_ms': [(440, 1820)], 'dur_s': 2.32, 'words': [], 'wav_path': 'C:/Users/druiv/Desktop/Jet_Files/Cloned_Repos/FireRedASR2S/assets/hello_zh.wav'}

result = asr_system.process("C:/Users/druiv/Desktop/Jet_Files/Cloned_Repos/FireRedASR2S/assets/hello_en.wav")
print(result)
# {'uttid': 'tmpid', 'text': 'Hello speech.', 'sentences': [{'start_ms': 260, 'end_ms': 1820, 'text': 'Hello speech.', 'asr_confidence': 0.933, 'lang': 'en', 'lang_confidence': 0.993}], 'vad_segments_ms': [(260, 1820)], 'dur_s': 2.24, 'words': [], 'wav_path': 'C:/Users/druiv/Desktop/Jet_Files/Cloned_Repos/FireRedASR2S/assets/hello_en.wav'}