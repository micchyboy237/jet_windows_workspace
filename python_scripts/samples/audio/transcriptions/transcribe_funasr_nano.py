from pathlib import Path

from funasr import AutoModel

BASE_DIR = Path(__file__).resolve().parent
REMOTE_CODE_PATH = BASE_DIR / "custom_model_fun_asr_nano.py"

# Single audio file to test all examples
SHORT_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_missav_20s.wav"
LONG_AUDIO = r"C:\Users\druiv\Desktop\Jet_Files\Mac_M1_Files\recording_spyx_1_speaker.wav"


def main(audio_path):
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=False,
        remote_code=str(REMOTE_CODE_PATH),
        device="cuda:0",
    )

    res = model.generate(
        input=[audio_path],
        cache={},
        batch_size=1,
        # hotwords=["开放时间"],
        # language="中文",
        language="ja",
        itn=True, # or False
    )
    text = res[0]["text"]
    print(text)

    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        remote_code=str(REMOTE_CODE_PATH),
        device="cuda:0",
    )
    res = model.generate(input=[audio_path], cache={}, batch_size=1)
    text = res[0]["text"]
    print(text)


if __name__ == "__main__":
    audio_path = SHORT_AUDIO
    # audio_path = LONG_AUDIO

    main(audio_path)